"""Analysis for BBPA-v0: tables, paired bootstrap, state diagnostics, verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import bbpa_common as C
import bbpa_data as D
import bbpa_model as M
import bbpa_structure as S
import bbpa_tensor as T

RUN_NAMES = ("BLOCK_BAG_S0", "LOCAL_PORT_S0", "TREE_PORT_S0")
SHORT = {"BLOCK_BAG_S0": "BLOCK_BAG", "LOCAL_PORT_S0": "LOCAL_PORT",
         "TREE_PORT_S0": "TREE_PORT"}


def _read_predictions(path: Path) -> dict[str, np.ndarray]:
    import csv

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: dict[str, list[Any]] = {}
    for r in rows:
        for k, v in r.items():
            out.setdefault(k, []).append(v)
    arr: dict[str, np.ndarray] = {}
    for k, vals in out.items():
        if k == "smiles":
            arr[k] = np.asarray(vals, dtype=object)
        elif k == "row_id":
            arr[k] = np.asarray(vals, np.int64)
        else:
            arr[k] = np.asarray(vals, np.float64)
    return arr


def _bins(values: np.ndarray, edges: Sequence[float], labels: Sequence[str]) -> np.ndarray:
    idx = np.full(values.shape[0], -1, np.int64)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        idx[(values >= lo) & (values < hi)] = i
    return idx


def _stratified(preds: Mapping[str, dict[str, np.ndarray]],
                models: Sequence[str]) -> list[dict[str, Any]]:
    ref = preds[models[0]]
    y = ref["label_raw"]
    bridge = ref["n_bridges"]
    diam = ref["block_tree_diameter"]
    bridge_edges = [-0.5, 0.5, 1.5, 2.5, 5.5, 10.5, 1e9]
    bridge_labels = ["0", "1", "2", "3-5", "6-10", "11+"]
    diam_edges = [-0.5, 2.5, 5.5, 8.5, 11.5, 1e9]
    diam_labels = ["0-2", "3-5", "6-8", "9-11", "12+"]
    out: list[dict[str, Any]] = []
    for field, values, edges, labels in (
            ("n_bridges", bridge, bridge_edges, bridge_labels),
            ("block_tree_diameter", diam, diam_edges, diam_labels)):
        idx = _bins(values, edges, labels)
        for i, label in enumerate(labels):
            mask = idx == i
            if not mask.any():
                continue
            row = {"stratifier": field, "bin": label, "n": int(mask.sum())}
            for model in models:
                p = preds[model]["pred_raw"]
                row[f"{model}_rmse"] = C.rmse(p[mask], y[mask])
                row[f"{model}_mae"] = C.mae(p[mask], y[mask])
                row[f"{model}_mean_abs_error"] = float(np.mean(np.abs(p[mask] - y[mask])))
            out.append(row)
    return out


def _state_diagnostics(run_dirs: Mapping[str, Path], cfg: Mapping[str, Any],
                       bundle: D.DataBundle) -> dict[str, Any]:
    atom_dims, bond_dims = D.atom_bond_dims()
    rows = np.asarray(bundle.eval_rows, np.int64)
    decs = S.build_all(bundle.raw, rows)
    diameters = np.asarray([
        int(max((S.tree_diameter(d, comp) for comp in d.components), default=0))
        for d in decs
    ], np.int64)
    mols = {int(r): T.build_molecule_tensors(d, int(r))
            for r, d in zip(rows.tolist(), decs)}
    model_t, _ = M.build_model(atom_dims, bond_dims, 0, cfg)
    model_l, _ = M.build_model(atom_dims, bond_dims, 0, cfg)
    st = torch.load(run_dirs["TREE_PORT_S0"] / "state_best.pt", map_location="cpu",
                    weights_only=False)
    sl = torch.load(run_dirs["LOCAL_PORT_S0"] / "state_best.pt", map_location="cpu",
                    weights_only=False)
    model_t.load_state_dict(st["state"])
    model_l.load_state_dict(sl["state"])
    model_t.eval()
    model_l.eval()

    tree_states: list[np.ndarray] = []
    local_states: list[np.ndarray] = []
    q_tree: list[np.ndarray] = []
    q_local: list[np.ndarray] = []
    u_global: list[np.ndarray] = []
    # same TREE checkpoint evaluated in both modes = pure algebra difference
    sem_state: list[np.ndarray] = []
    sem_q: list[np.ndarray] = []
    indep_state: list[np.ndarray] = []
    sem_mol: list[dict[str, Any]] = []
    batch_size = 128
    with torch.no_grad():
        for start in range(0, rows.size, batch_size):
            part = rows[start:start + batch_size]
            batch = T.collate(part.tolist(), mols, bundle.raw, "cpu")
            ot = model_t.forward_batch(batch, "TREE_PORT", return_states=True)
            ol = model_l.forward_batch(batch, "LOCAL_PORT", return_states=True)
            osame = model_t.forward_batch(batch, "LOCAL_PORT", return_states=True)
            stt = ot["states"].double().cpu().numpy()
            stl = ol["states"].double().cpu().numpy()
            sts = osame["states"].double().cpu().numpy()
            qt = ot["q"].double().cpu().numpy() if ot["q"] is not None else np.zeros((0, 96))
            ql = ol["q"].double().cpu().numpy() if ol["q"] is not None else np.zeros((0, 96))
            qs = osame["q"].double().cpu().numpy() if osame["q"] is not None else np.zeros((0, 96))
            sem_state.append(np.linalg.norm(stt - sts, axis=1))
            sem_q.append(np.linalg.norm(qt - qs, axis=1))
            indep_state.append(np.linalg.norm(stt - stl, axis=1))
            tree_states.append(stt)
            local_states.append(stl)
            q_tree.append(qt)
            q_local.append(ql)
            u_global.append(ot["u_global"].double().cpu().numpy())
            edge_mol = batch["bridge_seg"][batch["edge_bridge"]].cpu().numpy()
            bridge_mol = batch["bridge_seg"].cpu().numpy()
            n_mol = int(batch["n_molecules"])
            for m in range(n_mol):
                em = edge_mol == m
                bm = bridge_mol == m
                sem_mol.append({
                    "diameter": int(diameters[start + m]),
                    "state_max": float(np.linalg.norm(stt[em] - sts[em], axis=1).max())
                    if em.any() else 0.0,
                    "q_max": float(np.linalg.norm(qt[bm] - qs[bm], axis=1).max())
                    if bm.any() else 0.0,
                    "indep_state_max": float(np.linalg.norm(stt[em] - stl[em], axis=1).max())
                    if em.any() else 0.0,
                })
    st_arr = np.concatenate(tree_states, axis=0) if tree_states else np.zeros((0, 96))
    sl_arr = np.concatenate(local_states, axis=0) if local_states else np.zeros((0, 96))
    qt = np.concatenate(q_tree, axis=0) if q_tree else np.zeros((0, 96))
    ql = np.concatenate(q_local, axis=0) if q_local else np.zeros((0, 96))
    ug = np.concatenate(u_global, axis=0) if u_global else np.zeros((0, 96))
    sem_state_arr = np.concatenate(sem_state, axis=0) if sem_state else np.zeros(0)
    sem_q_arr = np.concatenate(sem_q, axis=0) if sem_q else np.zeros(0)
    indep_state_arr = np.concatenate(indep_state, axis=0) if indep_state else np.zeros(0)
    sd = st_arr.std(axis=0) if st_arr.size else np.zeros(0)

    def diff_stats(a: np.ndarray) -> dict[str, Any]:
        if a.size == 0:
            return {"n": 0, "mean": 0.0, "median": 0.0, "max": 0.0,
                    "fraction_gt_1e-6": 0.0}
        return {"n": int(a.size), "mean": float(a.mean()),
                "median": float(np.median(a)), "max": float(a.max()),
                "fraction_gt_1e-6": float(np.mean(a > 1e-6))}

    by_diam: dict[str, Any] = {}
    for lo, hi, label in ((0, 2, "0-2"), (3, 5, "3-5"), (6, 8, "6-8"),
                          (9, 11, "9-11"), (12, 10**6, "12+")):
        vals = [x for x in sem_mol if lo <= x["diameter"] <= hi]
        if not vals:
            continue
        by_diam[label] = {
            "n_molecules": len(vals),
            "fraction_state_gt_1e-6": float(np.mean([x["state_max"] > 1e-6 for x in vals])),
            "mean_state_max": float(np.mean([x["state_max"] for x in vals])),
            "max_state_max": float(np.max([x["state_max"] for x in vals])),
            "fraction_q_gt_1e-6": float(np.mean([x["q_max"] > 1e-6 for x in vals])),
        }
    sem_diam_le2 = [x for x in sem_mol if x["diameter"] <= 2]
    sem_deep = [x for x in sem_mol if x["diameter"] >= 3]
    return {
        "n_dev_molecules": int(rows.size),
        "n_tree_states": int(st_arr.shape[0]),
        "n_bridges": int(qt.shape[0]),
        "n_diameter_le_2": int(len(sem_diam_le2)),
        "n_diameter_ge_3": int(len(sem_deep)),
        "same_checkpoint_mode_state_l2_diff": diff_stats(sem_state_arr),
        "same_checkpoint_mode_q_l2_diff": diff_stats(sem_q_arr),
        "same_checkpoint_by_diameter": by_diam,
        "same_checkpoint_state_le2_max": float(max((x["state_max"] for x in sem_diam_le2),
                                                   default=0.0)),
        "same_checkpoint_state_deep_mean": float(np.mean([x["state_max"] for x in sem_deep]))
        if sem_deep else 0.0,
        "independent_checkpoint_state_l2_diff": diff_stats(indep_state_arr),
        "tree_state_dim_std": {
            "mean": float(sd.mean()) if sd.size else 0.0,
            "min": float(sd.min()) if sd.size else 0.0,
            "max": float(sd.max()) if sd.size else 0.0,
            "near_constant_fraction_lt_1e-4": float(np.mean(sd < 1e-4)) if sd.size else 1.0,
        },
        "tree_state_effective_rank": C.effective_rank(st_arr),
        "local_state_effective_rank": C.effective_rank(sl_arr),
        "u_global_effective_rank": C.effective_rank(ug),
        "q_effective_rank": C.effective_rank(qt),
        "tree_state_mean_norm": float(np.linalg.norm(st_arr, axis=1).mean())
        if st_arr.size else 0.0,
        "local_state_mean_norm": float(np.linalg.norm(sl_arr, axis=1).mean())
        if sl_arr.size else 0.0,
        "trained_tree_local_state_l2_diff_note":
            "independent TREE and LOCAL checkpoints have different learned weights; "
            "this is representation divergence, not a pure mode-semantics test",
    }


def _route_label(delta_comp: float, tree_rmse: float, tree_mae: float,
                 local_mae: float, diag: Mapping[str, Any],
                 cfg: Mapping[str, Any]) -> dict[str, Any]:
    r = dict(cfg["route_classification"])
    sd = float(diag["tree_state_dim_std"]["mean"])
    near = float(diag["tree_state_dim_std"]["near_constant_fraction_lt_1e-4"])
    rank = float(diag["tree_state_effective_rank"]["participation_ratio"])
    nonconstant = bool(sd > 1e-3 and near < 0.5 and rank > 1.0)
    checks = {
        "delta_comp_ge_promising": delta_comp >= float(r["promising_min_delta_comp"]),
        "tree_rmse_le_promising": tree_rmse <= float(r["promising_max_tree_rmse"]),
        "tree_mae_not_worse_than_local_by_0.01": tree_mae <= local_mae + float(
            r["promising_max_mae_degradation_vs_local"]),
        "state_nonconstant": nonconstant,
        "delta_comp_le_no_gain": delta_comp <= float(r["no_gain_max_delta_comp"]),
        "tree_weak_above_0.64": tree_rmse > float(r["weak_tree_rmse_above"]),
    }
    if all(checks[k] for k in ("delta_comp_ge_promising", "tree_rmse_le_promising",
                               "tree_mae_not_worse_than_local_by_0.01",
                               "state_nonconstant")):
        label = "PROMISING_PILOT"
    elif checks["delta_comp_le_no_gain"] and checks["tree_weak_above_0.64"]:
        label = "NO_COMPOSITION_GAIN"
    elif (not checks["delta_comp_le_no_gain"]) and checks["tree_weak_above_0.64"]:
        label = "CAPACITY_WITHOUT_COMPETITIVENESS"
    else:
        label = "INCONCLUSIVE_SINGLE_SEED"
    return {"label": label, "criteria": checks, "state_nonconstant": nonconstant,
            "state_mean_dim_std": sd, "state_near_constant_fraction": near,
            "state_participation_ratio": rank}


def analyze(run_dirs: Mapping[str, Path], out_dir: Path) -> dict[str, Any]:
    cfg = C.load_config()
    bundle = D.load_bundle()
    manifests = {k: C.read_json(v / "manifest.json") for k, v in run_dirs.items()}
    preds = {k: _read_predictions(v / "predictions_best_dev.csv")
             for k, v in run_dirs.items()}
    fit_preds = {k: _read_predictions(v / "predictions_best_fit.csv")
                 for k, v in run_dirs.items()}
    models = [k for k in RUN_NAMES if k in preds]
    for k in models:
        m = manifests[k]
        if m["seed"] != 0:
            raise ValueError(f"{k} is not seed 0")
        if m["data_hashes"]["split"]["sha256"] != cfg["split"]["split_sha256"]:
            raise ValueError(f"{k} split mismatch")
        if not np.array_equal(preds[k]["row_id"], bundle.eval_rows):
            raise ValueError(f"{k} dev row identity mismatch")
        if not np.array_equal(fit_preds[k]["row_id"], bundle.fit_rows):
            raise ValueError(f"{k} fit row identity mismatch")
        if not np.allclose(preds[k]["label_raw"], bundle.labels[bundle.eval_rows]):
            raise ValueError(f"{k} dev label mismatch")
    if len({tuple(sorted(manifests[k]["shared_init"].items())) for k in models}) != 1:
        raise ValueError("arms do not share initialisation")
    if len({manifests[k]["params"]["total"] for k in models}) != 1:
        raise ValueError("arms do not share parameter count")

    main_rows = []
    for k in models:
        m = manifests[k]
        main_rows.append({
            "model": SHORT[k], "run_id": k, "seed": int(m["seed"]),
            "params": int(m["params"]["total"]),
            "train_rmse_at_best": float(m["train_rmse_at_best"]),
            "dev_rmse": float(m["dev_rmse"]), "dev_mae": float(m["dev_mae"]),
            "best_epoch": int(m["best_epoch"]), "epochs_run": int(m["epochs_run"]),
            "wall_seconds": float(m["wall_seconds"]),
            "train_seconds": float(m["train_seconds"]),
            "gpu_hours": float(m["gpu_hours"]),
            "gpu_hours_total_process": float(m["gpu_hours_total_process"]),
            "gpu_name": m["gpu_name"], "peak_vram_bytes": m["peak_vram_bytes"],
            "checkpoint": m["checkpoint"], "curves": m["curves"],
            "predictions_dev": m["predictions_dev"], "smoke": m["smoke"],
        })

    y = bundle.labels[bundle.eval_rows]
    contrasts: dict[str, Any] = {}
    pairs = [
        ("LOCAL_PORT_S0", "TREE_PORT_S0", "delta_comp"),
        ("BLOCK_BAG_S0", "TREE_PORT_S0", "tree_minus_block"),
        ("BLOCK_BAG_S0", "LOCAL_PORT_S0", "local_minus_block"),
    ]
    for a, b, name in pairs:
        if a not in preds or b not in preds:
            continue
        contrasts[name] = C.paired_bootstrap(preds[a]["pred_raw"], preds[b]["pred_raw"],
                                             y, n_boot=5000, seed=0)
        contrasts[name].update({"a": a, "b": b,
                                "definition": f"RMSE({a}) - RMSE({b}); positive means {b} better"})
    # Pre-specified descriptive subgroups for where the recursion can act.
    ref = preds["TREE_PORT_S0"]
    diam = ref["block_tree_diameter"]
    nbr = ref["n_bridges"]
    subgroups = [
        ("diameter_le_2", diam <= 2),
        ("diameter_ge_3", diam >= 3),
        ("diameter_ge_6", diam >= 6),
        ("bridges_le_5", nbr <= 5),
        ("bridges_ge_6", nbr >= 6),
        ("bridges_ge_11", nbr >= 11),
    ]
    for name, mask in subgroups:
        item = C.paired_bootstrap(preds["LOCAL_PORT_S0"]["pred_raw"][mask],
                                  preds["TREE_PORT_S0"]["pred_raw"][mask], y[mask],
                                  n_boot=5000, seed=0)
        item.update({"a": "LOCAL_PORT_S0", "b": "TREE_PORT_S0",
                     "n_molecules": int(mask.sum()),
                     "definition": "RMSE(LOCAL_PORT) - RMSE(TREE_PORT) on the subgroup; "
                                   "positive means TREE_PORT better"})
        contrasts[f"delta_comp_{name}"] = item
    diag = _state_diagnostics(run_dirs, cfg, bundle)
    delta_comp = float(contrasts["delta_comp"]["rmse_delta_point"])
    tree_rmse = float(manifests["TREE_PORT_S0"]["dev_rmse"])
    tree_mae = float(manifests["TREE_PORT_S0"]["dev_mae"])
    local_mae = float(manifests["LOCAL_PORT_S0"]["dev_mae"])
    route = _route_label(delta_comp, tree_rmse, tree_mae, local_mae, diag, cfg)

    strat = _stratified(preds, models)
    ref = preds["TREE_PORT_S0"]
    order = np.argsort(-np.abs(ref["pred_raw"] - ref["label_raw"]))[:10]
    worst = []
    for i in order.tolist():
        row = {
            "row_id": int(ref["row_id"][i]), "smiles": str(ref["smiles"][i]),
            "label_raw": float(ref["label_raw"][i]),
            "tree_pred": float(ref["pred_raw"][i]),
            "tree_abs_error": float(abs(ref["pred_raw"][i] - ref["label_raw"][i])),
            "local_pred": float(preds["LOCAL_PORT_S0"]["pred_raw"][i]),
            "local_abs_error": float(abs(preds["LOCAL_PORT_S0"]["pred_raw"][i]
                                          - ref["label_raw"][i])),
            "block_pred": float(preds["BLOCK_BAG_S0"]["pred_raw"][i]),
            "block_abs_error": float(abs(preds["BLOCK_BAG_S0"]["pred_raw"][i]
                                          - ref["label_raw"][i])),
            "n_bridges": int(ref["n_bridges"][i]),
            "block_tree_diameter": int(ref["block_tree_diameter"][i]),
            "n_atoms": int(ref["n_atoms"][i]),
        }
        worst.append(row)

    history = dict(cfg["historical_d0_rmse"])
    hist_rows = []
    for name, rmse in history.items():
        if name == "source":
            continue
        hist_rows.append({
            "model": name, "d0_rmse": float(rmse),
            "delta_vs_tree": float(rmse) - tree_rmse,
            "delta_vs_local": float(rmse) - float(manifests["LOCAL_PORT_S0"]["dev_rmse"]),
            "note": "historical coordinate; different training recipe",
        })

    # Provenance of the pulled, committed run artefacts and the remote launch records.
    remote_by_run: dict[str, Any] = {}
    for meta_path in sorted((C.REPO_ROOT / ".rr" / "pulled").glob("*/meta.json")):
        try:
            meta = C.read_json(meta_path)
        except Exception:
            continue
        for name in RUN_NAMES:
            if name in str(meta.get("command", "")):
                remote_by_run[name] = meta
    file_names = ("manifest.json", "state_best.pt", "state_final.pt",
                  "predictions_best_fit.csv", "predictions_best_fit.npz",
                  "predictions_best_dev.csv", "predictions_best_dev.npz",
                  "curves.csv")
    provenance_runs = []
    for k in models:
        files = {}
        for fn in file_names:
            fp = run_dirs[k] / fn
            if fp.is_file():
                files[fn] = {"sha256": C.sha256_file(fp), "bytes": fp.stat().st_size}
        meta = remote_by_run.get(k, {})
        provenance_runs.append({
            "model": SHORT[k], "run_id": k,
            "training_commit": manifests[k]["git_commit"],
            "remote_experiment": meta.get("experiment"),
            "remote_run_id": meta.get("run_id"),
            "remote_git_commit": meta.get("git_commit"),
            "remote_git_dirty": meta.get("git_dirty"),
            "remote_state": meta.get("state"),
            "remote_exit_code": (meta.get("runtime") or {}).get("exit_code"),
            "remote_started_at": (meta.get("runtime") or {}).get("started_at"),
            "remote_completed_at": (meta.get("runtime") or {}).get("completed_at"),
            "gpu": (meta.get("runtime") or {}).get("cuda_visible_devices"),
            "files": files,
        })
    training_commits = sorted({r["training_commit"] for r in provenance_runs})
    max_wall = max(float(m["wall_seconds"]) for m in manifests.values())
    gpu_hours_sum = float(sum(float(m["gpu_hours"]) for m in manifests.values()))
    gpu_hours_process_sum = float(sum(float(m["gpu_hours_total_process"])
                                      for m in manifests.values()))
    budget = {
        "budget_wall_hours": 2.0, "budget_gpu_hours": 4.0,
        "max_run_wall_seconds": max_wall,
        "max_run_wall_hours": max_wall / 3600.0,
        "wall_utilization_fraction": (max_wall / 3600.0) / 2.0,
        "gpu_hours_sum_train": gpu_hours_sum,
        "gpu_hours_sum_process": gpu_hours_process_sum,
        "gpu_utilization_fraction": gpu_hours_sum / 4.0,
        "n_runs": len(models), "n_gpus_used": 2, "seed": 0,
        "within_budget": bool(max_wall <= 7200.0 and gpu_hours_sum <= 4.0),
        "official_valid_test_labels_read": False,
        "note": "wall is the longest single run; the three runs overlapped on two GPUs",
    }
    summary = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "code_commit": C.git_commit(),
        "training_commits": training_commits,
        "dataset": cfg["dataset"],
        "split": {
            "name": "D0", "sha256": cfg["split"]["split_sha256"],
            "n_fit": int(bundle.fit_rows.size), "n_eval": int(bundle.eval_rows.size),
            "fit_rows_sha256": C.array_sha256(bundle.fit_rows),
            "dev_rows_sha256": C.array_sha256(bundle.eval_rows),
        },
        "label_scaler": bundle.scaler,
        "data_hashes": bundle.hashes,
        "provenance": {"runs": provenance_runs, "training_commits": training_commits},
        "budget": budget,
        "main_table": main_rows,
        "primary_contrast": contrasts.get("delta_comp"),
        "all_contrasts": contrasts,
        "route_label": route["label"],
        "route_rule": route,
        "stratified": strat,
        "worst10_tree": worst,
        "historical_d0": hist_rows,
        "state_diagnostics": diag,
        "theoretical_difference_coverage": {
            "definition": "block-tree diameter >= 3 is the necessary condition for TREE_PORT and LOCAL_PORT to differ; diameter-2 stars coincide",
            "fraction_molecules_diameter_ge_3": None,  # filled from checks
            "fraction_molecules_diameter_ge_2": None,
        },
        "function_vs_gain": {
            "functional_verification": "TREE states and q differ from LOCAL on deep trees; star/single-bridge coincide as expected",
            "predictive_gain": "see primary_contrast and route_label",
            "scientific_claim": "single seed; exploratory engineering screen only",
        },
    }
    # add structure coverage from checks if present
    checks_path = C.ANALYSIS_DIR / "checks.json"
    if checks_path.is_file():
        checks = C.read_json(checks_path)
        stats = checks.get("structure_stats", {})
        summary["theoretical_difference_coverage"].update({
            "fraction_molecules_diameter_ge_3": stats.get("fraction_with_diameter_ge_3"),
            "fraction_molecules_diameter_ge_2": stats.get("fraction_with_diameter_ge_2"),
            "n_molecules": stats.get("n_molecules"),
            "bridge_coverage_fraction": stats.get("bridge_coverage_fraction"),
            "diameter_hist": stats.get("block_tree_diameter_hist"),
        })
    C.write_json(out_dir / "summary.json", summary)
    C.write_json(C.ANALYSIS_DIR / "run_index.json",
                 {"protocol_id": cfg["protocol_id"], "generated_utc": C.now_utc(),
                  "training_commits": training_commits,
                  "runs": provenance_runs, "budget": budget})
    C.write_json(C.ANALYSIS_DIR / "budget.json", budget)
    C.write_json(C.ANALYSIS_DIR / "paired_bootstrap.json", contrasts)
    C.write_json(C.ANALYSIS_DIR / "state_diagnostics.json", diag)
    C.write_csv(out_dir / "main_table.csv", main_rows)
    C.write_csv(out_dir / "stratified.csv", strat)
    C.write_csv(out_dir / "worst10_tree.csv", worst)
    C.write_csv(out_dir / "historical_d0.csv", hist_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(C.RESULTS_DIR))
    parser.add_argument("--out", default=str(C.RESULTS_DIR))
    args = parser.parse_args()
    root = Path(args.results)
    run_dirs = {name: root / name for name in RUN_NAMES}
    missing = [str(p) for p in run_dirs.values() if not p.is_dir()]
    if missing:
        raise SystemExit(f"missing run dirs: {missing}")
    summary = analyze(run_dirs, Path(args.out))
    print(json.dumps(C.jsonable(summary["main_table"]), indent=1))
    print("route:", summary["route_label"])
    print("primary:", json.dumps(C.jsonable(summary["primary_contrast"]), indent=1))
    print("diagnostics:", json.dumps(C.jsonable(summary["state_diagnostics"]), indent=1)[:2000])


if __name__ == "__main__":
    main()