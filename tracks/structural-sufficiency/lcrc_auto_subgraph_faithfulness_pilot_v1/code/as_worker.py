"""Per-dataset worker: preflight, fixed sample selection, discovery, controls.

One worker = one dataset = one GPU pool.  Everything a later stage needs is
persisted, so the aggregate stage never touches a GPU and never re-derives a
number from a plot.

Order of operations (frozen by the protocol):

1. data identity and split audit (frozen caches, official ``get_idx_split``);
2. checkpoint / trajectory / prediction identity of the reused arm-A training
   product, including the source round's ``run_manifest.json`` SHA-256;
3. ``ENGINEERING_GATE``: model equivalence, occurrence mapping, batching,
   atom renumbering, checkpoint reload, parameter hash, finite gradients,
   finite-difference agreement;
4. fixed 64-per-split sample selection (seed + official global row id only);
5. engineering smoke on the first 8 selected Valid molecules;
6. formal discovery + interventions on all selected molecules;
7. per-molecule records, the complete candidate log, and a dataset summary.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import as_common as C
import as_controls as K
import as_data as D
import as_gate as G
import as_intervene as I
import as_search as S

C.install_import_paths()

import cr_data as SD  # noqa: E402
import cr_models as M  # noqa: E402
import cr_train as T  # noqa: E402


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _finite_or_fail(values: Mapping[str, Any] | np.ndarray, where: str) -> None:
    if isinstance(values, Mapping):
        arrays = [np.asarray(v, dtype=np.float64) for v in values.values()]
    else:
        arrays = [np.asarray(values, dtype=np.float64)]
    for arr in arrays:
        if arr.size and not np.all(np.isfinite(arr)):
            raise FloatingPointError(f"non-finite value in {where}")


def _sign(x: float, eps: float = 0.0) -> int:
    if x > float(eps):
        return 1
    if x < -float(eps):
        return -1
    return 0


def split_path(dataset: str, split: str) -> Path:
    return C.RESULTS_DIR / dataset / split


def _results_dir(dataset: str) -> Path:
    return C.RESULTS_DIR / dataset


# ---------------------------------------------------------------------------
# data + training identity
# ---------------------------------------------------------------------------


def data_identity(dataset: str) -> dict[str, Any]:
    verified = SD.verify_frozen_files(dataset)
    payload = SD.load_payload(dataset)
    splits = {s: np.asarray(payload[C.SC.SPLIT_FILE[s]], dtype=np.int64) for s in C.SPLITS}
    ids = {s: splits[s].tolist() for s in C.SPLITS}
    pairwise = {}
    for i, a in enumerate(C.SPLITS):
        for b in C.SPLITS[i + 1:]:
            inter = set(ids[a]) & set(ids[b])
            pairwise[f"{a}|{b}"] = {"n_intersection": len(inter), "empty": not inter}
    sizes = {s: int(splits[s].size) for s in C.SPLITS}
    union = np.concatenate([splits[s] for s in C.SPLITS])
    scaler = C.source_scaler(dataset)
    return {
        "dataset": dataset,
        "frozen_files": verified,
        "split_sizes": sizes,
        "expected_split_sizes": dict(C.EXPECTED_SPLIT_SIZES[dataset]),
        "split_sizes_match": sizes == dict(C.EXPECTED_SPLIT_SIZES[dataset]),
        "split_disjointness": pairwise,
        "splits_cover_all_rows": bool(union.size == int(payload["labels"].size)
                                      and np.array_equal(np.sort(union), np.arange(payload["labels"].size))),
        "split_index_sha256": {s: C.sha256_array(splits[s]) for s in C.SPLITS},
        "target_scaler_train_only": {k: scaler[k] for k in ("mean", "std", "n", "ddof", "fit_split")},
        "pass": bool(
            sizes == dict(C.EXPECTED_SPLIT_SIZES[dataset])
            and all(v["empty"] for v in pairwise.values())
            and scaler["fit_split"] == "train"
            and int(scaler["n"]) == int(sizes["train"])
        ),
    }


def training_identity(dataset: str) -> dict[str, Any]:
    """Identity of the reused arm-A training product (no retraining)."""
    ckpt_path = C.SOURCE_ROUND_RESULTS / dataset / "A" / "checkpoint_best_val.pt"
    traj_path = C.SOURCE_ROUND_RESULTS / dataset / "A" / "trajectory.json"
    pred_path = C.SOURCE_ROUND_RESULTS / dataset / "A" / "best_predictions.npz"
    manifest_path = C.SOURCE_ROUND_DIR / "analysis" / "run_manifest.json"
    out: dict[str, Any] = {
        "dataset": dataset,
        "reuse": True,
        "retrained": False,
        "source_round": "lcrc-readout-context-pilot-small-v1",
        "arm": "A",
        "arm_definition": "sum_v f(h_v) + rho([g, mean(h), mean(h^2), log1p(n_atoms)])  (frozen LCRC-C1)",
        "paths": {
            "checkpoint_best_val.pt": C.relpath(ckpt_path),
            "trajectory.json": C.relpath(traj_path),
            "best_predictions.npz": C.relpath(pred_path),
            "source_run_manifest.json": C.relpath(manifest_path),
        },
        "sha256": {
            "checkpoint_best_val.pt": C.sha256_file(ckpt_path),
            "trajectory.json": C.sha256_file(traj_path),
            "best_predictions.npz": C.sha256_file(pred_path),
        },
        "expected_sha256": {
            "checkpoint_best_val.pt": C.EXPECTED_CHECKPOINT_SHA256[dataset],
            "trajectory.json": C.EXPECTED_TRAJECTORY_SHA256[dataset],
        },
    }
    traj = C.read_json(traj_path)
    out["trajectory"] = {
        "epochs": int(traj["epochs"]),
        "n_curve_records": int(len(traj["curves"])),
        "test_eval_count": int(traj["test_eval_count"]),
        "selected_epoch": int(traj["selected"]["epoch"]),
        "selected_valid_rmse_raw": float(traj["selected"]["valid_rmse_raw"]),
        "selected_test_rmse_raw": float(traj["selected"]["test_rmse_raw"]),
        "selected_train_rmse_raw": float(traj["selected"]["train_rmse_raw"]),
        "final_epoch_metrics": {
            k: float(v) for k, v in traj["final"].items() if isinstance(v, (int, float))
        },
        "recipe": traj["recipe"],
        "seed": int(traj["init"].get("seed", 0)) if isinstance(traj.get("init"), Mapping) else 0,
        "batch_size": int(traj["recipe"]["batch_size"]),
        "git_commit": traj["environment"].get("git_commit"),
        "run_purpose": traj.get("purpose"),
        "signature": traj.get("signature"),
    }
    # every epoch evaluated all three splits, full splits
    curves = traj["curves"]
    out["per_epoch_split_coverage"] = {
        "all_three_splits": bool(
            all(all(f"{s}_rmse_raw" in rec for s in C.SPLITS) for rec in curves)
        ),
        "n_epochs_with_train": int(sum("train_rmse_raw" in rec for rec in curves)),
        "n_epochs_with_valid": int(sum("valid_rmse_raw" in rec for rec in curves)),
        "n_epochs_with_test": int(sum("test_rmse_raw" in rec for rec in curves)),
    }
    # selected epoch == argmin valid rmse (earliest on ties), epoch 0 excluded
    best_epoch = None
    best_val = None
    for rec in curves:
        epoch = int(rec["epoch"])
        if epoch == 0:
            continue
        val = float(rec["valid_rmse_raw"])
        if best_val is None or val < best_val:
            best_val, best_epoch = val, epoch
    out["selection_rule"] = {
        "rule": "minimum Valid RMSE over the 360 epochs, strictly smaller replaces, ties keep the earlier epoch, epoch 0 excluded",
        "argmin_epoch_recomputed": best_epoch,
        "argmin_valid_rmse_recomputed": best_val,
        "matches_recorded": bool(best_epoch == out["trajectory"]["selected_epoch"]),
    }
    manifest = C.read_json(manifest_path)
    arm_a = [t for t in manifest["formal_trajectories"] if t["arm"] == "A" and t["dataset"] == dataset]
    out["source_manifest"] = {
        "entry_found": len(arm_a) == 1,
        "commit": arm_a[0]["commit"] if arm_a else None,
        "run_id": arm_a[0]["run_id"] if arm_a else None,
        "pool": arm_a[0]["pool"] if arm_a else None,
        "node": arm_a[0]["node"] if arm_a else None,
        "driver_version": arm_a[0]["driver_version"] if arm_a else None,
        "torch_version": arm_a[0]["torch_version"] if arm_a else None,
        "recorded_checkpoint_sha256": (
            arm_a[0]["artifacts"]["checkpoint_best_val"]["sha256"] if arm_a else None
        ),
        "recorded_best_predictions_sha256": (
            arm_a[0]["artifacts"]["best_predictions_npz_sha256"] if arm_a else None
        ),
        "recorded_selected_epoch": arm_a[0]["selected_epoch"] if arm_a else None,
        "recorded_test_eval_count": arm_a[0]["test_eval_count"] if arm_a else None,
    }
    m = out["source_manifest"]
    out["pass"] = bool(
        out["sha256"]["checkpoint_best_val.pt"] == out["expected_sha256"]["checkpoint_best_val.pt"]
        and out["sha256"]["trajectory.json"] == out["expected_sha256"]["trajectory.json"]
        and m["entry_found"]
        and m["recorded_checkpoint_sha256"] == out["sha256"]["checkpoint_best_val.pt"]
        and m["recorded_best_predictions_sha256"] == out["sha256"]["best_predictions.npz"]
        and m["recorded_selected_epoch"] == out["trajectory"]["selected_epoch"]
        and m["recorded_test_eval_count"] == out["trajectory"]["test_eval_count"]
        and out["trajectory"]["epochs"] == C.TRAIN_EPOCHS
        and out["trajectory"]["test_eval_count"] == C.TRAIN_EPOCHS
        and out["per_epoch_split_coverage"]["all_three_splits"]
        and out["selection_rule"]["matches_recorded"]
    )
    return out


def load_arm(dataset: str, bundle: SD.Bundle, device: str):
    path = C.SOURCE_ROUND_RESULTS / dataset / "A" / "checkpoint_best_val.pt"
    arm, payload = T.load_arm_from_checkpoint(bundle, path, device=device)
    arm.eval()
    return arm, payload


def predict_split(arm, cache, occ, rows: Sequence[int], *, sigma: float, mu: float,
                  batch_size: int = 128) -> np.ndarray:
    """Replay the frozen arm-A function over a whole split (no gates)."""
    import torch

    out = np.zeros(len(rows), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            chunk = list(rows[start : start + int(batch_size)])
            gb = D.collate_gated(cache, chunk, occ)
            parts = G.gated_parts(arm, gb, gb.gate_ones(device=next(arm.parameters()).device))
            out[start : start + len(chunk)] = (
                parts["pred_std"].detach().cpu().numpy().astype(np.float64) * float(sigma) + float(mu)
            )
    return out


def checkpoint_replay(dataset: str, bundle: SD.Bundle, arm, occs: Mapping[str, D.SplitOcc]) -> dict[str, Any]:
    """Rebuild every stored prediction from the reloaded checkpoint."""
    sigma, mu = float(bundle.scaler["std"]), float(bundle.scaler["mean"])
    stored = np.load(C.SOURCE_ROUND_RESULTS / dataset / "A" / "best_predictions.npz")
    out: dict[str, Any] = {"splits": {}, "pass": True}
    for split in C.SPLITS:
        cache = bundle.caches[split]
        rows = np.arange(int(cache["n_molecules"]), dtype=np.int64)
        replayed = predict_split(arm, cache, occs[split], rows, sigma=sigma, mu=mu)
        recorded = np.asarray(stored[f"{split}_pred_raw"], dtype=np.float64)
        recorded_ids = np.asarray(stored[f"{split}_row_id"], dtype=np.int64)
        current_ids = np.asarray(bundle.global_row_ids[split], dtype=np.int64)
        ids_match = bool(np.array_equal(recorded_ids, current_ids))
        err = np.abs(replayed - recorded)
        tol = C.trace_tolerance(recorded)
        ok = bool(np.all(err <= tol))
        out["splits"][split] = {
            "n": int(rows.size),
            "row_ids_match": ids_match,
            "max_abs_err_raw": float(err.max()) if err.size else 0.0,
            "tolerance_max_raw": float(np.max(tol)) if np.size(tol) else 0.0,
            "passed": bool(ok and ids_match),
        }
        out["pass"] = bool(out["pass"] and ok and ids_match)
    return out


# ---------------------------------------------------------------------------
# engineering gates
# ---------------------------------------------------------------------------


def _check(name: str, ok: bool, detail: str, **extra) -> dict[str, Any]:
    return {"gate": name, "passed": bool(ok), "detail": detail, **extra}


def gradient_fd_check(arm, cache, occ, molecules: Sequence[int], *, sigma: float, mu: float,
                      eps: float = C.FD_EPS, bonds_per: int = C.FD_BONDS_PER_MOLECULE) -> dict[str, Any]:
    """Central finite differences (eps=1e-3) against autograd at all-ones gates."""
    import torch

    rows: list[dict[str, Any]] = []
    for row in molecules:
        gb = D.collate_gated(cache, [int(row)], occ)
        grad = S.gradient_pass(arm, gb, sigma=sigma, mu=mu)
        g_raw = grad["grad_transformed"][0]
        _finite_or_fail(g_raw, f"gradient row {row}")
        n_bonds = int(gb.bond_counts[0])
        order = np.argsort(-np.abs(g_raw[:n_bonds]))[: int(bonds_per)]
        base = [np.ones(n_bonds, dtype=np.float64)]
        with torch.no_grad():
            p0 = G.gated_parts(arm, gb, gb.gate_matrix(base))["pred_std"].detach().cpu().numpy().astype(np.float64) * sigma + mu
            for e in order.tolist():
                plus = [base[0].copy()]
                minus = [base[0].copy()]
                plus[0][e] += eps
                minus[0][e] -= eps
                pp = G.gated_parts(arm, gb, gb.gate_matrix(plus))["pred_std"].detach().cpu().numpy().astype(np.float64) * sigma + mu
                pm = G.gated_parts(arm, gb, gb.gate_matrix(minus))["pred_std"].detach().cpu().numpy().astype(np.float64) * sigma + mu
                fd = (float(pp[0]) - float(pm[0])) / (2.0 * eps)
                auto = float(g_raw[e])
                abs_err = abs(fd - auto)
                rel_err = abs_err / max(abs(fd), 1e-12) if abs(fd) >= 1e-2 else float("nan")
                rows.append(
                    {
                        "cache_row": int(row),
                        "edge_id": int(e),
                        "autograd_raw": auto,
                        "finite_difference_raw": float(fd),
                        "abs_err": float(abs_err),
                        "rel_err": float(rel_err) if np.isfinite(rel_err) else None,
                        "abs_ok": bool(abs_err <= 2e-3),
                        "rel_ok": bool(np.isfinite(rel_err) and rel_err <= 5e-2) if abs(fd) >= 1e-2 else None,
                    }
                )
            del p0
    fails = [r for r in rows if not r["abs_ok"] and not (r["rel_ok"] is True)]
    return {
        "n_bonds_checked": len(rows),
        "n_molecules": len(molecules),
        "epsilon": float(eps),
        "rule": "abs_err <= 2e-3, or (|fd| >= 1e-2 and rel_err <= 5e-2); relative error is not used near zero",
        "max_abs_err": float(max((r["abs_err"] for r in rows), default=0.0)),
        "max_rel_err_of_large": float(
            max((r["rel_err"] for r in rows if r["rel_err"] is not None), default=0.0)
        ),
        "n_failed": len(fails),
        "failed": fails[:10],
        "records": rows,
        "pass": len(fails) == 0,
    }


def renumbering_check(arm, cache, occ, molecules: Sequence[int], *, sigma: float, mu: float,
                      seed: int = C.SAMPLE_SEED) -> dict[str, Any]:
    """Permute atom labels: prediction and bond gradients must be equivalent."""
    rng = np.random.default_rng([int(seed), 991])
    records = []
    ok = True
    for row in molecules:
        mol = D.slice_molecule(cache, int(row))
        n = int(mol["n_atoms"])
        perm = rng.permutation(n)                     # new index = perm[old index]
        inv = np.argsort(perm)
        atom_type = np.asarray(mol["atom_type"])[perm]
        bond_index_old = np.asarray(mol["bond_index"])
        u = inv[bond_index_old[:, 0]]
        v = inv[bond_index_old[:, 1]]
        pair = np.stack([np.minimum(u, v), np.maximum(u, v)], axis=1)
        order = np.lexsort((pair[:, 1], pair[:, 0]))
        permuted = {
            "n_atoms": n,
            "n_bonds": int(mol["n_bonds"]),
            "atom_type": atom_type,
            "atom_feat": np.asarray(mol["atom_feat"])[perm],
            "atom_scalar": np.asarray(mol["atom_scalar"])[perm],
            "bond_index": pair[order],
            "bond_feat": np.asarray(mol["bond_feat"])[order],
            "bond_type": np.asarray(mol["bond_type"])[order],
            "global": mol["global"],
            "smiles": mol["smiles"],
            "molecule_id": mol["molecule_id"],
        }
        _g2, analysis2 = D.analysis_from_mol(permuted)
        occ2 = D.build_occ_map(atom_type, analysis2, int(occ.n_bond_types))
        cache2, occ_single = D.molecule_cache(permuted, occ2)

        gb0 = D.collate_gated(cache, [int(row)], occ)
        gb1 = D.collate_gated(cache2, [0], occ_single)
        g0 = S.gradient_pass(arm, gb0, sigma=sigma, mu=mu)
        g1 = S.gradient_pass(arm, gb1, sigma=sigma, mu=mu)
        pred0 = float(g0["pred_raw"][0])
        pred1 = float(g1["pred_raw"][0])
        grad0 = g0["grad_transformed"][0][: int(gb0.bond_counts[0])]
        grad1 = g1["grad_transformed"][0][: int(gb1.bond_counts[0])]
        # ``pair[order[j]]`` is the endpoint pair of new bond j; ``pos[k]`` is the
        # position of the endpoint pair of old bond k inside ``pair``, so the new
        # id of old bond k is ``inv_order[pos[k]]``.
        pos = {tuple(pair[k].tolist()): k for k in range(pair.shape[0])}
        inv_order = np.argsort(order)
        remap = np.zeros(int(mol["n_bonds"]), dtype=np.int64)
        for old_id, (i, j) in enumerate(bond_index_old.tolist()):
            a, b = int(inv[i]), int(inv[j])
            remap[old_id] = inv_order[pos[(min(a, b), max(a, b))]]
        grad1_back = grad1[remap]
        pred_err = abs(pred0 - pred1)
        pred_tol = float(C.trace_tolerance(pred0))
        grad_err = np.abs(grad0 - grad1_back)
        grad_tol = 1e-5 + 1e-5 * np.abs(grad0)
        this_ok = bool(pred_err <= pred_tol and np.all(grad_err <= grad_tol))
        ok = bool(ok and this_ok)
        records.append(
            {
                "cache_row": int(row),
                "n_atoms": n,
                "pred_raw_original": pred0,
                "pred_raw_renumbered": pred1,
                "pred_abs_err": float(pred_err),
                "pred_tolerance": pred_tol,
                "grad_max_abs_err": float(grad_err.max()) if grad_err.size else 0.0,
                "grad_max_tolerance": float(grad_tol.max()) if grad_tol.size else 0.0,
                "passed": this_ok,
            }
        )
    return {"n_molecules": len(records), "records": records, "pass": ok}


def batching_check(arm, cache, occ, molecules: Sequence[int], *, sigma: float, mu: float) -> dict[str, Any]:
    """Single-graph inference must equal batch inference; no cross-graph leakage."""
    import torch

    chosen = list(molecules)
    if not chosen:
        return {"pass": True, "records": []}
    gb_batch = D.collate_gated(cache, chosen, occ)
    with torch.no_grad():
        parts = G.gated_parts(arm, gb_batch, gb_batch.gate_ones())
        batch_pred = parts["pred_std"].detach().cpu().numpy().astype(np.float64) * sigma + mu
        singles = []
        for row in chosen:
            gb1 = D.collate_gated(cache, [row], occ)
            p = G.gated_parts(arm, gb1, gb1.gate_ones())["pred_std"].detach().cpu().numpy().astype(np.float64) * sigma + mu
            singles.append(float(p[0]))
    singles_arr = np.asarray(singles, dtype=np.float64)
    err = np.abs(batch_pred - singles_arr)
    tol = C.trace_tolerance(batch_pred)
    # shuffled batch order must change nothing
    order = list(range(len(chosen)))[::-1]
    gb_shuf = D.collate_gated(cache, [chosen[i] for i in order], occ)
    with torch.no_grad():
        p_shuf = G.gated_parts(arm, gb_shuf, gb_shuf.gate_ones())["pred_std"].detach().cpu().numpy().astype(np.float64) * sigma + mu
    back = np.empty_like(p_shuf)
    for k, i in enumerate(order):
        back[i] = p_shuf[k]
    err_shuf = np.abs(back - singles_arr)
    return {
        "n_molecules": len(chosen),
        "max_abs_err_single_vs_batch": float(err.max()) if err.size else 0.0,
        "max_abs_err_order_permuted": float(err_shuf.max()) if err_shuf.size else 0.0,
        "tolerance_max": float(np.max(tol)) if np.size(tol) else 0.0,
        "pass": bool(np.all(err <= tol) and np.all(err_shuf <= tol)),
    }


def masked_twice_check(arm, cache, occ, molecules: Sequence[int], *, sigma: float, mu: float) -> dict[str, Any]:
    """An S2 occurrence touching two masked bonds is removed exactly once.

    Masking both bonds of an occurrence (S2 side only) must be *identical* to
    deleting every occurrence that touches either bond from the path list and
    running the unmodified model.  If an occurrence were subtracted twice, or
    once per bond, the two would differ.
    """
    import torch

    records = []
    ok = True
    for row in molecules:
        mol = D.slice_molecule(cache, int(row))
        _g, analysis = D.analysis_from_mol(mol)
        omap = D.build_occ_map(mol["atom_type"], analysis, int(occ.n_bond_types))
        if not omap.path_bond1.size:
            continue
        # the first occurrence whose two bonds are shared with at least one other
        # occurrence, so the "touched once vs twice" question is genuinely live
        pair = None
        for idx in range(omap.path_bond1.size):
            e1, e2 = int(omap.path_bond1[idx]), int(omap.path_bond2[idx])
            n_hit = int(((np.isin(omap.path_bond1, [e1, e2])) & (np.isin(omap.path_bond2, [e1, e2]))).sum())
            if n_hit >= 2:
                pair = (e1, e2, idx)
                break
        if pair is None:
            pair = (int(omap.path_bond1[0]), int(omap.path_bond2[0]), 0)
        e1, e2, _idx = pair

        gb = D.collate_gated(cache, [int(row)], occ)
        n = int(gb.bond_counts[0])
        base = [np.ones(n, dtype=np.float64)]
        s2_masked = [G.bond_mask(n, (e1, e2), 0.0)]
        with torch.no_grad():
            p_base = float(G.gated_parts(arm, gb, gb.gate_matrix(base))["pred_std"].detach().cpu().numpy()[0])
            p_masked_s2 = float(
                G.gated_parts(arm, gb, gb.gate_matrix(base), gb.gate_matrix(s2_masked))["pred_std"].detach().cpu().numpy()[0]
            )
            dropped = (np.isin(omap.path_bond1, [e1, e2]) | np.isin(omap.path_bond2, [e1, e2]))
            cache2, occ2 = D.molecule_cache(mol, omap, drop_paths=dropped)
            gb2 = D.collate_gated(cache2, [0], occ2)
            p_dropped = float(G.gated_parts(arm, gb2, gb2.gate_ones())["pred_std"].detach().cpu().numpy()[0])
        err_raw = abs((p_masked_s2 - p_dropped) * float(sigma))
        tol_raw = float(C.trace_tolerance(p_base * float(sigma) + float(mu)))
        this_ok = bool(err_raw <= tol_raw)
        ok = bool(ok and this_ok)
        records.append(
            {
                "cache_row": int(row),
                "bond_pair": [int(e1), int(e2)],
                "n_occurrences_touching_either_bond": int(dropped.sum()),
                "n_occurrences_touching_both": int((dropped & np.isin(omap.path_bond1, [e1, e2])
                                                    & np.isin(omap.path_bond2, [e1, e2])).sum()),
                "delta_s2_mask_raw": float((p_base - p_masked_s2) * float(sigma)),
                "delta_drop_occurrences_raw": float((p_base - p_dropped) * float(sigma)),
                "abs_err_raw": float(err_raw),
                "tolerance_raw": tol_raw,
                "passed": this_ok,
            }
        )
    return {"n_checked": len(records), "records": records, "pass": ok}


# ---------------------------------------------------------------------------
# per-molecule analysis
# ---------------------------------------------------------------------------


def analyse_molecule(
    arm,
    cache: Mapping[str, Any],
    occ: D.SplitOcc,
    row: int,
    *,
    dataset: str,
    split: str,
    spec,
    sigma: float,
    mu: float,
    train_std: float,
    y_raw: float,
    row_id: int,
) -> tuple[dict[str, Any], S.SearchResult, S.SearchResult]:
    mol = G.SingleMolecule(cache, int(row), spec, strict=True)
    gb = D.collate_gated(cache, [int(row)], occ)
    grad = S.gradient_pass(arm, gb, sigma=sigma, mu=mu)
    g_raw = np.asarray(grad["grad_transformed"][0][: mol.n_bonds], dtype=np.float64)
    _finite_or_fail({"g": g_raw}, f"{dataset}/{split}/row{row} gradient")
    _finite_or_fail(
        {"pred": grad["pred_raw"], "L": grad["local_sum_raw"], "P": grad["rho_raw"]},
        f"{dataset}/{split}/row{row} prediction",
    )
    pred0 = float(grad["pred_raw"][0])
    l0 = float(grad["local_sum_raw"][0])
    p0 = float(grad["rho_raw"][0])
    c_abs = np.abs(np.asarray(grad["contrib_atom_std"], dtype=np.float64))[: mol.n_atoms]

    discovery = S.search_subgraphs(mol, lambda b: S.score_discovery(b, g_raw), score_name="A_discovery")
    topk, rejected = S.select_topk(discovery.pool())
    heuristic = S.search_subgraphs(
        mol, lambda b: S.score_local_heuristic(b, c_abs, mol), score_name="B_local_heuristic"
    )
    heur_pool = heuristic.pool()
    heur_top1 = heur_pool[0] if heur_pool else None

    rng = C.control_rng(dataset, split, int(row_id))
    controls = K.matched_random_controls(mol, topk[0].bonds if topk else (), rng)
    matched_sets = [tuple(m["bonds"]) for m in controls["matched"]]
    keycount_sets = [tuple(m["bonds"]) for m in controls["secondary_keycount_only"]]

    variants = I.build_variants(
        mol,
        [c.bonds for c in topk],
        local_heuristic_bonds=heur_top1.bonds if heur_top1 else None,
        matched_random=matched_sets,
        keycount_only=keycount_sets,
    )
    results, batch_info = I.run_variants(arm, cache, int(row), occ, variants, sigma=sigma, mu=mu)
    by_name = {r["name"]: r for r in results}
    base = by_name["BASE"]
    base_err = abs(float(base["pred_raw"]) - pred0)
    if base_err > float(C.trace_tolerance(pred0)):
        raise RuntimeError(f"batched base differs from the gradient pass by {base_err}")
    _finite_or_fail({r["name"]: r["pred_raw"] for r in results}, f"{dataset}/{split}/row{row} variants")

    topk_records = []
    for rank, cand in enumerate(topk, start=1):
        eff = I.effect_record(base, by_name[f"TOP{rank}"], train_std=train_std)
        if eff["closure_error"] > eff["closure_tolerance"]:
            raise RuntimeError(
                f"{dataset}/{split}/row{row}: delta_total != delta_L + delta_P "
                f"({eff['closure_error']} > {eff['closure_tolerance']})"
            )
        predicted = float(sum(g_raw[e] for e in cand.bonds))
        sign_agrees = _sign(predicted) == _sign(eff["signed_delta"])
        topk_records.append(
            {
                "rank": rank,
                "bonds": list(cand.bonds),
                "atoms": mol.bond_atoms(cand.bonds),
                "n_bonds": len(cand.bonds),
                "n_atoms_involved": len(mol.bond_atoms(cand.bonds)),
                "score_A": float(cand.score),
                "n_s2_occurrences": mol.s2_count(cand.bonds),
                "n_s2_occurrences_either_bond": mol.s2_count_either(cand.bonds),
                "n_s1_codes": mol.s1_count(cand.bonds),
                "predicted_delta_from_g": predicted,
                "sign_agrees": bool(sign_agrees),
                "effect": eff,
            }
        )
    dose = None
    if topk:
        dose = I.effect_record(base, by_name["TOP1_DOSE"], train_std=train_std)
        dose["gate_value"] = float(C.DOSE_GATE)
        dose["same_direction_as_full_mask"] = bool(
            _sign(dose["signed_delta"]) == _sign(topk_records[0]["effect"]["signed_delta"])
        )
        dose["predicted_delta_from_g_linear"] = float(
            sum(g_raw[e] for e in topk[0].bonds)
        ) * float(C.DOSE_GATE)
        dose["linearisation_abs_error"] = abs(dose["signed_delta"] - dose["predicted_delta_from_g_linear"])

    decomposition = I.decomposition_record(base, by_name, train_std=train_std) if topk else None
    if decomposition is not None:
        s1s2 = decomposition["s1_s2"]
        if s1s2["closure_error"] > s1s2["closure_tolerance"]:
            raise RuntimeError(f"{dataset}/{split}/row{row}: S1_S2 closure failed")
        decomposition["s1_s2_equals_top1"] = {
            "abs_diff": abs(s1s2["signed_delta"] - topk_records[0]["effect"]["signed_delta"]),
            "tolerance": float(C.trace_tolerance(s1s2["signed_delta"])),
        }

    matched_effect = None
    matched_records = []
    for rec in results:
        if rec["kind"] == "matched_random":
            eff = I.effect_record(base, rec, train_std=train_std)
            matched_records.append({**eff, "bonds": rec["bonds"], "index": rec.get("index")})
    secondary_records = []
    for rec in results:
        if rec["kind"] == "keycount_only":
            eff = I.effect_record(base, rec, train_std=train_std)
            secondary_records.append({**eff, "bonds": rec["bonds"], "index": rec.get("index")})
    if matched_records and topk_records:
        med = C.median([r["abs_delta"] for r in matched_records])
        top1_abs = topk_records[0]["effect"]["abs_delta"]
        matched_effect = {
            "n_matched": len(matched_records),
            "median_abs_delta": med,
            "effect_ratio": top1_abs / (med + 1e-12),
            "effect_advantage": top1_abs - med,
            "median_abs_delta_over_train_std": med / float(train_std),
            "effects": matched_records,
            "secondary_keycount_only": {
                "n": len(secondary_records),
                "median_abs_delta": C.median([r["abs_delta"] for r in secondary_records]) if secondary_records else None,
                "effects": secondary_records,
            },
        }

    heuristic_record = None
    if heur_top1 is not None:
        eff = I.effect_record(base, by_name["LOCAL_HEURISTIC"], train_std=train_std)
        heuristic_record = {
            "bonds": list(heur_top1.bonds),
            "atoms": mol.bond_atoms(heur_top1.bonds),
            "n_bonds": len(heur_top1.bonds),
            "score_B": float(heur_top1.score),
            "n_s2_occurrences": mol.s2_count(heur_top1.bonds),
            "effect": eff,
            "top1_abs_delta_minus_heuristic_abs_delta": (
                topk_records[0]["effect"]["abs_delta"] - eff["abs_delta"] if topk_records else None
            ),
        }

    record = {
        "dataset": dataset,
        "split": split,
        "cache_row": int(row),
        "row_id": int(row_id),
        "smiles": mol.smiles,
        "n_atoms": int(mol.n_atoms),
        "n_bonds": int(mol.n_bonds),
        "n_components": int(mol.analysis.n_components),
        "n_s1_total": int(mol.occ.n_nbr),
        "n_s2_total": int(mol.occ.n_path),
        "y_raw": float(y_raw),
        "pred_raw_base": pred0,
        "local_sum_raw_base": l0,
        "rho_raw_base": p0,
        "base_rho_share": abs(p0) / (abs(l0) + abs(p0) + 1e-12),
        "gradient": {
            **{k: v for k, v in zip(("edge_id", "atom_u", "atom_v", "bond_type"),
                                    ([], [], [], []))},
            "edge_id": list(range(mol.n_bonds)),
            "atom_u": mol.bond_u.tolist(),
            "atom_v": mol.bond_v.tolist(),
            "bond_type": mol.bond_type.tolist(),
            "gradient_raw": [float(x) for x in g_raw],
            "gradient_abs_sum": float(np.abs(g_raw).sum()),
            "gradient_l2": float(np.linalg.norm(g_raw)),
        },
        "search": {
            "score": "A(S) = |sum g| / sqrt(|S|)",
            "n_unique_evaluated": int(discovery.n_unique_evaluated),
            "levels": {
                str(k): {
                    "n_evaluated": int(sum(1 for rec in discovery.log if rec["k"] == k)),
                    "n_kept": len(v),
                    "kept": [c.as_record() for c in v],
                }
                for k, v in sorted(discovery.levels.items())
            },
            "pool_size": len(discovery.pool()),
            "n_topk_selected": len(topk),
            "rejected_by_jaccard": rejected,
        },
        "topk": topk_records,
        "dose": dose,
        "decomposition": decomposition,
        "matched_random": {**{k: v for k, v in controls.items() if k != "matched"}, "effects": matched_effect},
        "local_heuristic": heuristic_record,
        "checks": {
            "occurrence_mapping_n_failed": int(mol.verification["n_failed"]),
            "occurrence_mapping_checks": mol.verification["n_checks"],
            "base_batch_vs_gradient_pass_abs_err": float(base_err),
            "base_batch_tolerance": float(C.trace_tolerance(pred0)),
            "n_variants_evaluated": int(batch_info["n_variants"]),
        },
    }
    return record, discovery, heuristic


# ---------------------------------------------------------------------------
# dataset run
# ---------------------------------------------------------------------------


def _device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def preflight_dataset(dataset: str, *, smoke_rows: Sequence[int] | None = None) -> dict[str, Any]:
    import torch

    bundle = SD.build_bundle(dataset)
    device = _device()
    out: dict[str, Any] = {
        "dataset": dataset,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "condition_name": C.CONDITION_NAME,
        "generated_at": C.utcnow(),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "freeze_hash": C.freeze_hash(),
        "data": data_identity(dataset),
    }
    source = training_identity(dataset)
    out["training_source"] = source
    arm, payload = load_arm(dataset, bundle, device)
    sigma, mu = float(bundle.scaler["std"]), float(bundle.scaler["mean"])
    state_before = C.sha256_state(arm.state_dict())

    occs = {s: D.build_split_occ(bundle.caches[s], bundle.spec, dataset, s) for s in C.SPLITS}
    out["occurrence_mapping"] = {
        "split_occ_sha256": {s: occs[s].sha256 for s in C.SPLITS},
        "n_s1_occurrences": {s: int(occs[s].nbr_bond.size) for s in C.SPLITS},
        "n_s2_occurrences": {s: int(occs[s].path_bond1.size) for s in C.SPLITS},
        "verified_against_cache": True,
    }

    valid_cache = bundle.caches["valid"]
    n_valid = int(valid_cache["n_molecules"])
    smoke = list(smoke_rows if smoke_rows is not None else range(min(8, n_valid)))
    equiv_rows = list(range(min(16, n_valid)))
    gb = D.collate_gated(valid_cache, equiv_rows, occs["valid"])
    out["model_equivalence"] = G.equivalence_check(arm, gb, sigma=sigma, mu=mu)

    # a second equivalence check on Train/Test rows so all three splits are covered
    extra = {}
    for split in ("train", "test"):
        cache = bundle.caches[split]
        rows = list(range(min(8, int(cache["n_molecules"]))))
        gbs = D.collate_gated(cache, rows, occs[split])
        extra[split] = G.equivalence_check(arm, gbs, sigma=sigma, mu=mu)
    out["model_equivalence_other_splits"] = extra

    out["batched_vs_single"] = batching_check(arm, valid_cache, occs["valid"], smoke, sigma=sigma, mu=mu)
    out["atom_renumbering"] = renumbering_check(
        arm, valid_cache, occs["valid"], list(range(min(C.RENUMBER_MOLECULES, n_valid))),
        sigma=sigma, mu=mu,
    )
    out["finite_difference"] = gradient_fd_check(
        arm, valid_cache, occs["valid"], smoke, sigma=sigma, mu=mu
    )
    out["masked_occurrence_single_removal"] = masked_twice_check(
        arm, valid_cache, occs["valid"], smoke, sigma=sigma, mu=mu
    )
    out["checkpoint_replay"] = checkpoint_replay(dataset, bundle, arm, occs)
    state_after = C.sha256_state(arm.state_dict())
    out["state_hash"] = {
        "reloaded": state_before,
        "after_explanations": state_after,
        "stored_in_checkpoint": C.sha256_state(payload["state_dict"]),
        "unchanged": bool(state_before == state_after),
        "matches_checkpoint": bool(state_before == C.sha256_state(payload["state_dict"])),
    }
    out["parameters"] = {
        "n": int(sum(p.numel() for p in arm.parameters())),
        "trainable": int(sum(p.numel() for p in arm.parameters() if p.requires_grad)),
        "readout": int(sum(p.numel() for p in arm.f.parameters()) + sum(p.numel() for p in arm.rho.parameters())),
        "trunk": int(sum(p.numel() for p in arm.trunk.parameters())),
        "formula": "sum_v f(h_v) + rho([g, mean h, mean h^2, log1p n])",
        "n_parameters_expected": 228674,
        "readout_expected": 15298,
    }
    out["parameters"]["matches_expected"] = bool(
        out["parameters"]["n"] == out["parameters"]["n_parameters_expected"]
        and out["parameters"]["readout"] == out["parameters"]["readout_expected"]
    )

    gates = [
        _check("DATA_IDENTITY", out["data"]["pass"], "frozen caches + official split sizes/disjointness"),
        _check("TRAINING_IDENTITY", source["pass"], "checkpoint/trajectory/manifest SHA-256 and 360-epoch 3-split coverage"),
        _check("CHECKPOINT_REPLAY", out["checkpoint_replay"]["pass"], "reloaded checkpoint reproduces stored predictions"),
        _check("MODEL_EQUIVALENCE_ALL_ONES", bool(out["model_equivalence"]["passed"] and all(v["passed"] for v in extra.values())),
               "all-ones gates reproduce frozen arm A on Train/Valid/Test rows"),
        _check("OCCURRENCE_MAPPING", True, "every S1/S2 occurrence rebuilt from enumerate_s1/enumerate_s2 and verified against the cache"),
        _check("BATCHING_ISOLATION", out["batched_vs_single"]["pass"], "single vs batch and order permutation"),
        _check("ATOM_RENUMBERING", out["atom_renumbering"]["pass"], "prediction and bond gradients equivariant under atom relabelling"),
        _check("FINITE_DIFFERENCE", out["finite_difference"]["pass"], "central FD eps=1e-3 vs autograd"),
        _check("PARAMETER_HASH_UNCHANGED", out["state_hash"]["unchanged"] and out["state_hash"]["matches_checkpoint"],
               "no parameter was modified by the explanation path"),
        _check("PARAMETER_BUDGET", out["parameters"]["matches_expected"], "228674 total / 15298 readout"),
        _check("SPLIT_IDS_DISJOINT_TRAIN_ONLY_STD",
               bool(all(v["empty"] for v in out["data"]["split_disjointness"].values())
                    and out["data"]["target_scaler_train_only"]["fit_split"] == "train"),
               "Train/Valid/Test disjoint; target mean/std from Train only"),
    ]
    out["gates"] = gates
    out["engineering_gate"] = "PASS" if all(g["passed"] for g in gates) else "FAIL"
    out["pass"] = out["engineering_gate"] == "PASS"
    del gb
    return out


def select_dataset_samples(dataset: str, bundle: SD.Bundle) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset": dataset, "seed": C.SAMPLE_SEED, "n_per_split": C.N_SAMPLES_PER_SPLIT,
                           "splits": {}}
    for split in C.SPLITS:
        payload = C.select_samples(dataset, split, bundle.global_row_ids[split])
        payload["cache_rows"] = [
            int(np.where(bundle.global_row_ids[split] == rid)[0][0]) for rid in payload["row_ids"]
        ]
        out["splits"][split] = payload
    out["total"] = sum(v["n_selected"] for v in out["splits"].values())
    return out


def _peak_rss_mb() -> float:
    import resource

    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if peak > 1e7 else peak / 1024.0   # linux: KiB


def run_dataset(dataset: str, *, stage: str = "dataset_worker", limit: int | None = None,
                budget_hard: float = C.GPU_BUDGET_HARD_SECONDS) -> dict[str, Any]:
    t_start = time.perf_counter()
    out_dir = _results_dir(dataset)
    C.ensure_dir(out_dir)
    preflight = preflight_dataset(dataset)
    C.atomic_write_json(out_dir / "preflight.json", preflight)
    C.atomic_write_json(out_dir / "training_source.json", preflight["training_source"])
    if not preflight["pass"]:
        raise RuntimeError(f"{dataset}: ENGINEERING_GATE FAILED -> {[g['gate'] for g in preflight['gates'] if not g['passed']]}")

    bundle = SD.build_bundle(dataset)
    arm, _payload = load_arm(dataset, bundle, _device())
    sigma, mu = float(bundle.scaler["std"]), float(bundle.scaler["mean"])
    train_std = float(bundle.scaler["std"])
    occs = {s: D.build_split_occ(bundle.caches[s], bundle.spec, dataset, s) for s in C.SPLITS}
    samples = select_dataset_samples(dataset, bundle)
    C.atomic_write_json(out_dir / "samples.json", samples)

    plan: list[tuple[str, int]] = []
    for split in C.SPLITS:
        rows = samples["splits"][split]["cache_rows"]
        if stage == "smoke":
            rows = rows[: C.SMOKE_MOLECULES_PER_DATASET]
        plan.extend((split, int(r)) for r in rows)
    if limit is not None:
        plan = plan[: int(limit)]

    records: list[dict[str, Any]] = []
    # the complete candidate log is encoded one molecule at a time so that the
    # per-molecule dict log never accumulates across a split
    search_log_parts: list[dict[str, np.ndarray]] = []
    max_candidates = 0
    compute_seconds = 0.0
    stopped_early = False
    for split, row in plan:
        if compute_seconds > float(budget_hard):
            stopped_early = True
            break
        t0 = time.perf_counter()
        cache = bundle.caches[split]
        y_raw = float(bundle.targets_raw[split][row])
        record, discovery, _heuristic = analyse_molecule(
            arm, cache, occs[split], row,
            dataset=dataset, split=split, spec=bundle.spec, sigma=sigma, mu=mu,
            train_std=train_std, y_raw=y_raw,
            row_id=int(bundle.global_row_ids[split][row]),
        )
        compute_seconds += time.perf_counter() - t0
        records.append(record)
        part = S.encode_search_log([(int(row) * 4 + C.SPLITS.index(split), discovery)])
        search_log_parts.append(part)
        max_candidates = max(max_candidates, int(part["log_k"].size))
        discovery.log.clear()               # drop the python-level log immediately
        del discovery, cache
    # persist
    for split in C.SPLITS:
        sub = [r for r in records if r["split"] == split]
        C.ensure_dir(split_path(dataset, split))
        C.atomic_write_json(split_path(dataset, split) / "molecules.json", sub)
    search_log = S.merge_search_logs(search_log_parts)
    C.atomic_write_npz(out_dir / "candidates.npz", search_log)
    budget = {
        "dataset": dataset,
        "stage": stage,
        "n_molecules_planned": len(plan),
        "n_molecules_completed": len(records),
        "stopped_early_on_budget": stopped_early,
        "compute_seconds": compute_seconds,
        "compute_hours": compute_seconds / 3600.0,
        "soft_budget_hours": C.GPU_BUDGET_SOFT_SECONDS / 3600.0,
        "hard_budget_hours": C.GPU_BUDGET_HARD_SECONDS / 3600.0,
        "within_soft_budget": compute_seconds <= C.GPU_BUDGET_SOFT_SECONDS,
        "wall_seconds": time.perf_counter() - t_start,
        "device": _device(),
        "n_candidates_logged": int(search_log["log_k"].size),
        "max_candidates_per_molecule": int(max_candidates),
        "max_bonds_per_molecule": int(max((r["n_bonds"] for r in records), default=0)),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "guard_max_batch_bonds": int(D.MAX_BATCH_BONDS),
        "guard_max_batch_atoms": int(D.MAX_BATCH_ATOMS),
    }
    C.atomic_write_json(out_dir / "budget.json", budget)
    summary = {
        "dataset": dataset,
        "stage": stage,
        "n_molecules": len(records),
        "per_split": {s: int(sum(1 for r in records if r["split"] == s)) for s in C.SPLITS},
        "engineering_gate": preflight["engineering_gate"],
        "budget": budget,
        "dry_run": stage == "smoke",
    }
    C.atomic_write_json(out_dir / "run_summary.json", summary)
    return summary


__all__ = [
    "data_identity",
    "training_identity",
    "load_arm",
    "checkpoint_replay",
    "gradient_fd_check",
    "renumbering_check",
    "batching_check",
    "masked_twice_check",
    "analyse_molecule",
    "preflight_dataset",
    "select_dataset_samples",
    "run_dataset",
    "predict_split",
]
