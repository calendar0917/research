"""GWIC-v0 analysis: main table, paired bootstrap, word/coverage diagnostics.

Reads only committed run artefacts.  Dev labels are used for evaluation and
descriptive bootstrap; they never enter training or checkpoint selection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import gwic_common as C
import gwic_data as D
import gwic_model as M
import gwic_structure as S
import gwic_tensor as T

ARMS = ("SENTENCE_BASE", "WORD_GEOM_NULL", "WORD_REAL")
RUN_DIRS = {a: C.RESULTS_DIR / f"{a}_S0" for a in ARMS}


def _norm(bundle: D.DataBundle) -> dict[str, np.ndarray]:
    return {
        "atom_scalar_mean": bundle.atom_scalar_mean,
        "atom_scalar_scale": bundle.atom_scalar_scale,
        "graph_mean": bundle.graph_mean,
        "graph_scale": bundle.graph_scale,
    }


def _load_model(bundle: D.DataBundle, arm: str, device: Any = "cpu"):
    run = RUN_DIRS[arm]
    manifest = C.read_json(run / "manifest.json")
    ckpt = torch.load(run / "state_best.pt", map_location=device, weights_only=False)
    atom_dims, bond_dims = D.atom_bond_dims()
    model, info = M.build_model(atom_dims, bond_dims, int(manifest["seed"]), _norm(bundle),
                                use_words=bool(arm != "SENTENCE_BASE"),
                                config=manifest.get("config"))
    model.load_state_dict(ckpt["state"])
    model.to(device)
    model.eval()
    return model, manifest, ckpt


def _predict(model: M.GwicModel, structs, raw, rows: Sequence[int], arm: str,
             scaler: Mapping[str, float], device: Any, batch_size: int = 256,
             use_geometry: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    out_raw = np.zeros(len(rows), np.float64)
    out_std = np.zeros(len(rows), np.float64)
    rows = np.asarray(rows, np.int64)
    with torch.no_grad():
        for start in range(0, rows.size, batch_size):
            part = rows[start:start + batch_size]
            batch = T.collate(part.tolist(), structs, raw, device)
            y = model(batch, arm=arm, use_geometry=use_geometry)["yhat_std"].cpu().numpy()
            out_std[start:start + part.size] = y
            out_raw[start:start + part.size] = y * float(scaler["std"]) + float(scaler["mean"])
    return out_raw, out_std


def _paired_bootstrap(y: np.ndarray, pred: Mapping[str, np.ndarray],
                      n_boot: int = 5000, seed: int = 12345) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = int(y.size)
    arms = list(pred.keys())
    sq = {a: (pred[a] - y) ** 2 for a in arms}
    ae = {a: np.abs(pred[a] - y) for a in arms}
    # point estimates
    rmse = {a: float(np.sqrt(sq[a].mean())) for a in arms}
    mae = {a: float(ae[a].mean()) for a in arms}
    stats = {"delta_word": [], "delta_binding": [], "delta_content": [],
             "delta_word_mae": [], "delta_binding_mae": [], "delta_content_mae": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        r = {a: float(np.sqrt(sq[a][idx].mean())) for a in arms}
        m = {a: float(ae[a][idx].mean()) for a in arms}
        stats["delta_word"].append(r["SENTENCE_BASE"] - r["WORD_REAL"])
        stats["delta_binding"].append(r["WORD_GEOM_NULL"] - r["WORD_REAL"])
        stats["delta_content"].append(r["SENTENCE_BASE"] - r["WORD_GEOM_NULL"])
        stats["delta_word_mae"].append(m["SENTENCE_BASE"] - m["WORD_REAL"])
        stats["delta_binding_mae"].append(m["WORD_GEOM_NULL"] - m["WORD_REAL"])
        stats["delta_content_mae"].append(m["SENTENCE_BASE"] - m["WORD_GEOM_NULL"])
    def ci(vals):
        arr = np.asarray(vals, np.float64)
        return {"point": float(arr.mean()), "median": float(np.median(arr)),
                "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
                "n_boot": int(n_boot)}
    return {
        "n_molecules": n,
        "rmse": rmse, "mae": mae,
        "delta_word": {"point": rmse["SENTENCE_BASE"] - rmse["WORD_REAL"],
                       "bootstrap": ci(stats["delta_word"])},
        "delta_binding": {"point": rmse["WORD_GEOM_NULL"] - rmse["WORD_REAL"],
                          "bootstrap": ci(stats["delta_binding"])},
        "delta_content": {"point": rmse["SENTENCE_BASE"] - rmse["WORD_GEOM_NULL"],
                          "bootstrap": ci(stats["delta_content"])},
        "delta_word_mae": {"point": mae["SENTENCE_BASE"] - mae["WORD_REAL"],
                           "bootstrap": ci(stats["delta_word_mae"])},
        "delta_binding_mae": {"point": mae["WORD_GEOM_NULL"] - mae["WORD_REAL"],
                              "bootstrap": ci(stats["delta_binding_mae"])},
        "delta_content_mae": {"point": mae["SENTENCE_BASE"] - mae["WORD_GEOM_NULL"],
                              "bootstrap": ci(stats["delta_content_mae"])},
        "seed": int(seed),
    }


def _effective_rank(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return {"participation_ratio": 0.0, "entropy_rank": 0.0, "top1_share": 0.0,
                "mean_dim_std": 0.0, "min_dim_std": 0.0, "near_constant_fraction": 1.0}
    x = x - x.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(1, x.shape[0] - 1)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    total = ev.sum()
    if total <= 1e-20:
        return {"participation_ratio": 0.0, "entropy_rank": 0.0, "top1_share": 1.0,
                "mean_dim_std": 0.0, "min_dim_std": 0.0, "near_constant_fraction": 1.0}
    p = ev / total
    pr = float(total * total / np.sum(ev * ev))
    ent = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))))
    std = x.std(axis=0)
    return {"participation_ratio": pr, "entropy_rank": ent,
            "top1_share": float(ev.max() / total),
            "mean_dim_std": float(std.mean()), "min_dim_std": float(std.min()),
            "near_constant_fraction": float((std < 1e-4).mean())}


def _diagnostics(bundle: D.DataBundle, structs, raw, arm: str, model: M.GwicModel,
                 device: Any) -> dict[str, Any]:
    rows = np.asarray(bundle.eval_rows, np.int64)
    w_all, q_all, z_all = [], [], []
    alpha_entropies, alpha_eff, alpha_const, alpha_stds = [], [], [], []
    top_sizes, all_sizes, weighted_sizes = [], [], np.zeros(5, np.float64)
    top_rows: list[dict[str, Any]] = []
    max_alpha_by_root = []
    with torch.no_grad():
        for start in range(0, rows.size, 128):
            part = rows[start:start + 128]
            batch = T.collate(part.tolist(), structs, raw, device)
            out = model(batch, arm=arm, return_aux=True)
            w = out["word"]["w"].cpu().numpy()
            alpha = out["alpha"].cpu().numpy()
            q = out["q"].cpu().numpy()
            z = out["z"].cpu().numpy()
            w_all.append(w); q_all.append(q); z_all.append(z)
            ptr = batch["root_word_ptr"].cpu().numpy()
            support = batch["word_support"].cpu().numpy()
            roots = int(batch["n_atoms"])
            for r in range(roots):
                lo, hi = int(ptr[r]), int(ptr[r + 1])
                if hi <= lo:
                    continue
                a = alpha[lo:hi]
                a = a / max(1e-12, a.sum())
                h = float(-np.sum(a * np.log(np.maximum(a, 1e-12))))
                alpha_entropies.append(h)
                alpha_eff.append(float(np.exp(h)))
                alpha_const.append(bool(float(a.max() - a.min()) < 1e-6) and hi - lo > 1)
                alpha_stds.append(float(a.std()))
                max_alpha_by_root.append(float(a.max()))
                k = int(np.argmax(a))
                sup = [int(x) for x in support[lo + k].tolist() if x >= 0]
                top_sizes.append(len(sup))
                all_sizes.extend([int((support[j] >= 0).sum()) for j in range(lo, hi)])
                for j in range(lo, hi):
                    sz = int((support[j] >= 0).sum())
                    weighted_sizes[sz] += float(a[j - lo])
                mol_ptr = batch["mol_atom_ptr"].cpu().numpy()
                mol = int(np.searchsorted(mol_ptr, int(batch["word_root"][lo + k]), side="right") - 1)
                mol_start = int(mol_ptr[mol])
                root_local = int(batch["word_root"][lo + k]) - mol_start
                atom_feat = batch["atom_feat"].cpu().numpy()
                bond_feat = batch["bond_feat"].cpu().numpy()
                wcand = int(lo + k)
                sup_global = [int(x) for x in support[wcand].tolist() if x >= 0]
                ib = [int(x) for x in batch["word_internal"][wcand].tolist() if x >= 0]
                top_rows.append({
                    "row_id": int(part[mol]),
                    "smiles": str(np.asarray(raw["smiles"])[int(part[mol])]),
                    "root_local": root_local,
                    "root_atom_type": int(atom_feat[mol_start + root_local, 0]),
                    "support_local": [int(x) - mol_start for x in sup_global],
                    "support_atom_types": [int(atom_feat[x, 0]) for x in sup_global],
                    "internal_bond_types": [[int(v) for v in bond_feat[x].tolist()] for x in ib],
                    "support_size": len(sup_global),
                    "boundary_bond_count": int(round(float(np.expm1(float(batch["word_scalars"][wcand, 11].item()))))),
                    "alpha_top": float(a[k]),
                    "alpha_entropy": h,
                    "n_candidates_root": hi - lo,
                })
    w_cat = np.concatenate(w_all, axis=0) if w_all else np.zeros((0, 64))
    q_cat = np.concatenate(q_all, axis=0) if q_all else np.zeros((0, 64))
    z_cat = np.concatenate(z_all, axis=0) if z_all else np.zeros((0, 64))
    n_roots = int(len(alpha_entropies))
    all_sizes = np.asarray(all_sizes, np.float64)
    weighted_sizes = weighted_sizes / max(1e-12, weighted_sizes.sum())
    top_rows = sorted(top_rows, key=lambda x: (-x["alpha_top"], x["row_id"], x["root_local"]))[:20]
    # sampled pairwise cosine of word representations, descriptive only
    cos_mean = 0.0
    if w_cat.shape[0] >= 2:
        rng = np.random.default_rng(17)
        k = min(1000, w_cat.shape[0])
        sample = w_cat[rng.choice(w_cat.shape[0], size=k, replace=False)]
        sample = sample / np.maximum(1e-12, np.linalg.norm(sample, axis=1, keepdims=True))
        cos = sample @ sample.T
        off = cos[~np.eye(k, dtype=bool)]
        cos_mean = float(off.mean()) if off.size else 0.0
    return {
        "alpha_std_mean": float(np.mean(alpha_stds)) if n_roots else 0.0,
        "word_pairwise_cosine_mean_sampled": cos_mean,
        "n_roots_with_candidates": n_roots,
        "alpha_entropy_mean": float(np.mean(alpha_entropies)) if n_roots else 0.0,
        "alpha_entropy_median": float(np.median(alpha_entropies)) if n_roots else 0.0,
        "alpha_effective_candidates_mean": float(np.mean(alpha_eff)) if n_roots else 0.0,
        "alpha_effective_candidates_median": float(np.median(alpha_eff)) if n_roots else 0.0,
        "alpha_constant_fraction": float(np.mean(alpha_const)) if n_roots else 0.0,
        "alpha_top_mean": float(np.mean(max_alpha_by_root)) if n_roots else 0.0,
        "top_candidate_size_distribution": {str(k): int(sum(1 for s in top_sizes if s == k))
                                            for k in (2, 3, 4)},
        "all_candidate_size_distribution": {
            str(k): int(np.sum(all_sizes == k)) for k in (2, 3, 4)},
        "alpha_weighted_candidate_size_distribution": {
            str(k): float(weighted_sizes[k]) for k in (2, 3, 4)},
        "word_rank": _effective_rank(w_cat),
        "q_rank": _effective_rank(q_cat),
        "z_rank": _effective_rank(z_cat),
        "word_rows": int(w_cat.shape[0]),
        "q_rows": int(q_cat.shape[0]),
        "high_alpha_examples": top_rows,
    }


def _degenerate(diag: Mapping[str, Any]) -> bool:
    return bool(diag["word_rank"]["participation_ratio"] < 2.0
                or diag["q_rank"]["participation_ratio"] < 2.0
                or diag["alpha_constant_fraction"] > 0.90
                or diag["z_rank"]["mean_dim_std"] < 1e-6)


def main() -> dict[str, Any]:
    bundle = D.load_bundle()
    cfg = C.load_config()
    rows = np.asarray(bundle.eval_rows, np.int64)
    y = np.asarray(bundle.labels[rows], np.float64)
    structs = S.build_all(bundle.raw, rows)
    device = "cpu"
    preds: dict[str, np.ndarray] = {}
    preds_std: dict[str, np.ndarray] = {}
    main_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    switched: dict[str, Any] = {}
    for arm in ARMS:
        model, manifest, ckpt = _load_model(bundle, arm, device)
        p_raw, p_std = _predict(model, structs, bundle.raw, rows, arm, bundle.scaler, device)
        preds[arm] = p_raw
        preds_std[arm] = p_std
        main_rows.append({
            "arm": arm,
            "params": int(manifest["params"]["total"]),
            "train_rmse_at_best": float(manifest["train_rmse_at_best"]),
            "dev_rmse": float(C.rmse(p_raw, y)),
            "dev_mae": float(C.mae(p_raw, y)),
            "best_epoch": int(manifest["best_epoch"]),
            "epochs_run": int(manifest["epochs_run"]),
            "wall": float(manifest["wall_seconds"]),
            "gpu_hours": float(manifest["gpu_hours"]),
            "prep_seconds": float(manifest["prep_seconds"]),
            "train_seconds": float(manifest["train_seconds"]),
            "training_commit": manifest.get("git_commit"),
        })
        if arm != "SENTENCE_BASE":
            diagnostics[arm] = _diagnostics(bundle, structs, bundle.raw, arm, model, device)
            p_zero, _ = _predict(model, structs, bundle.raw, rows, arm, bundle.scaler,
                                 device, use_geometry=False)
            p_real, _ = _predict(model, structs, bundle.raw, rows, arm, bundle.scaler,
                                 device, use_geometry=True)
            switched[arm] = {
                "rmse_geometry_zero": float(C.rmse(p_zero, y)),
                "rmse_geometry_real": float(C.rmse(p_real, y)),
                "mae_geometry_zero": float(C.mae(p_zero, y)),
                "mae_geometry_real": float(C.mae(p_real, y)),
                "max_abs_prediction_change": float(np.max(np.abs(p_real - p_zero))),
                "mean_abs_prediction_change": float(np.mean(np.abs(p_real - p_zero))),
                "fraction_changed_gt_1e_5": float(np.mean(np.abs(p_real - p_zero) > 1e-5)),
            }
    boot = _paired_bootstrap(y, preds)
    table = {r["arm"]: r for r in main_rows}
    delta_word = table["SENTENCE_BASE"]["dev_rmse"] - table["WORD_REAL"]["dev_rmse"]
    delta_binding = table["WORD_GEOM_NULL"]["dev_rmse"] - table["WORD_REAL"]["dev_rmse"]
    degen = _degenerate(diagnostics["WORD_REAL"])
    rule = cfg["route_classification"]
    if (delta_word >= float(rule["promising_min_word_gain_vs_base"])
            and delta_binding >= float(rule["promising_min_binding_gain_vs_null"])
            and table["WORD_REAL"]["dev_rmse"] <= float(rule["promising_max_word_dev_rmse"])
            and not degen):
        verdict = "PROMISING_WORD_ROUTE"
    elif (delta_word >= float(rule["word_content_min_gain_vs_base"])
          and delta_binding < float(rule["clear_binding_min_gain"])):
        verdict = "WORD_CONTENT_ONLY"
    elif (delta_word <= 0.0 or (table["WORD_REAL"]["dev_rmse"] > float(rule["ridge_reference_rmse"])
                                and delta_binding < float(rule["clear_binding_min_gain"]))):
        verdict = "NO_COMPETITIVE_WORD_SIGNAL"
    else:
        verdict = "INCONCLUSIVE_SINGLE_SEED"

    summary = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "analysis_commit": C.git_commit(),
        "training_commits": sorted({str(r["training_commit"]) for r in main_rows}),
        "seed": 0,
        "data": {
            "dataset": cfg["dataset"], "split": cfg["split"],
            "scaler": bundle.scaler,
            "n_fit": int(bundle.fit_rows.size), "n_dev": int(bundle.eval_rows.size),
            "raw_cache_sha256": bundle.hashes["raw_cache"]["sha256"],
            "split_sha256": bundle.hashes["split"]["sha256"],
        },
        "arms": table,
        "main_table": main_rows,
        "contrasts": {
            "delta_word": delta_word,
            "delta_binding": delta_binding,
            "bootstrap": boot,
            "definition": "positive = WORD_REAL better",
        },
        "diagnostics": diagnostics,
        "checkpoint_geometry_switch": switched,
        "degenerate_word_real": degen,
        "verdict": verdict,
        "route_rule": rule,
        "historical_d0_rmse": cfg["historical_d0_rmse"],
    }
    C.write_json(C.RESULTS_DIR / "summary.json", summary)
    C.write_json(C.RESULTS_DIR / "paired_bootstrap.json", boot)
    C.write_json(C.RESULTS_DIR / "diagnostics.json", diagnostics)
    C.write_csv(C.RESULTS_DIR / "main_table.csv", main_rows)
    for arm in ARMS:
        np.savez(C.RESULTS_DIR / f"predictions_{arm}_S0.npz",
                 row_id=rows, pred=preds[arm], pred_std=preds_std[arm], label=y)
    print(json.dumps(C.jsonable(summary), indent=1, sort_keys=True)[:8000])
    return summary


if __name__ == "__main__":
    main()
