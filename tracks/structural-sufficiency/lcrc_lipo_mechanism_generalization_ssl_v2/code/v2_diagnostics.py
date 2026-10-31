"""Inference-only diagnostics for the v2 round.

No training and no Monitor label is used here.  The perturbations are applied
to the *dev* fold of each internal split, so all labels are Fit2352 labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import v2_common as C
import v2_data as D
import v2_folds as F
import v2_models as M


def _load_model(key: str):
    payload = C.read_json(C.RUNS / key / "train.json")
    if payload.get("monitor_labels_read") is not False:
        raise PermissionError("diagnostic source run is not label-blind")
    spec = C.load_frozen_spec()
    norm = C.load_frozen_norm()
    recipe_name = str(payload["recipe"])
    if recipe_name not in C.RECIPES:
        raise ValueError(f"unknown recipe in diagnostic source run: {recipe_name}")
    model, _meta = M.build_model(
        payload["arm"], spec, norm, int(payload["seed"]), "cpu", dropout=float(C.RECIPES[recipe_name]["dropout"])
    )
    state = torch.load(C.RUNS / key / "soup.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def _dev_cache_and_rows(fold: int):
    cache = C.load_fit_cache()["cache"]
    folds = F.load_folds()
    rows = [int(r) for r in folds["folds"][fold]["rows_global"]]
    dev_cache = D.select_cache(cache, F.fold_positions(folds, fold), split_label=f"fold{fold}_dev_diag")
    return dev_cache, rows


def _predict_batch(model, batch, target_norm) -> np.ndarray:
    with torch.no_grad():
        z = model(batch).detach().cpu().numpy().reshape(-1)
    return D.denormalise(z, target_norm)


def _permute_segments(batch: Mapping[str, Any], keys: Sequence[str], *, same_perm: bool, seed: int) -> dict[str, Any]:
    out = dict(batch)
    path_atom = batch["path_atom"]
    if path_atom.numel() == 0:
        return out
    g = torch.Generator().manual_seed(int(seed))
    order = torch.argsort(path_atom, stable=True)
    atoms = path_atom[order]
    # Segment boundaries in the sorted order.
    unique, counts = torch.unique_consecutive(atoms, return_counts=True)
    assert int(unique.numel()) == int(torch.unique(path_atom).numel())
    offset = 0
    per_key = {k: batch[k].clone() for k in keys}
    for count in counts.tolist():
        idx = order[offset : offset + count]
        if count > 1:
            if same_perm:
                perm = torch.randperm(count, generator=g)
                for k in keys:
                    per_key[k][idx] = batch[k][idx][perm]
            else:
                for k in keys:
                    perm = torch.randperm(count, generator=g)
                    per_key[k][idx] = batch[k][idx][perm]
        offset += count
    out.update(per_key)
    return out


def _global_train_mean(batch: Mapping[str, Any], mean: np.ndarray) -> dict[str, Any]:
    out = dict(batch)
    out["global"] = torch.as_tensor(np.asarray(mean, dtype=np.float32), dtype=torch.float32).repeat(
        int(batch["n_graphs"]), 1
    )
    return out


def _error_structure(fold: int, key: str, cache) -> dict[str, Any]:
    folds = F.load_folds()
    positions = [int(p) for p in F.fold_positions(folds, fold)]
    rows = [int(r) for r in F.fold_rows(folds, fold)]
    data = np.load(C.RUNS / key / "predictions.npz")
    pred = np.asarray(data["pred"], dtype=np.float64)
    if [int(x) for x in data["rows"].tolist()] != positions:
        raise ValueError("diagnostic prediction row order mismatch")
    fit = C.load_fit_labels()
    y_all = np.asarray(fit["y"], dtype=np.float64)
    y = y_all[np.asarray(positions, dtype=np.int64)]
    err = pred - y
    abs_err = np.abs(err)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    n_atoms = np.diff(atom_ptr).astype(np.float64)
    feats = {
        "n_atoms": n_atoms,
        "n_components": np.asarray(cache["n_components"], dtype=np.float64),
        "n_junctions": np.asarray(cache["n_junctions"], dtype=np.float64),
        "n_ring_systems": np.asarray(cache["n_ring_systems"], dtype=np.float64),
    }
    names = [f.name for f in C.load_frozen_spec().global_feature_fields]
    for name in (
        "log1p_n_bonds",
        "cyclomatic_number",
        "ring_atom_fraction",
        "aromatic_atom_fraction",
        "max_degree",
        "mean_formal_charge",
    ):
        feats[name] = np.asarray(cache["global"], dtype=np.float64)[:, names.index(name)]
    correlations = {}
    for name, v in feats.items():
        if np.std(v) == 0:
            correlations[name] = {"pearson_abs_error": None, "spearman_abs_error": None}
            continue
        correlations[name] = {
            "pearson_abs_error": float(np.corrcoef(v, abs_err)[0, 1]),
            "spearman_abs_error": float(
                np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(abs_err)))[0, 1]
            ),
        }
    q = np.quantile(n_atoms, [0.25, 0.5, 0.75])
    bins = np.digitize(n_atoms, q, right=True)
    by_size = []
    for b in range(4):
        m = bins == b
        if not m.any():
            continue
        by_size.append(
            {
                "bin": b,
                "atom_range": [float(n_atoms[m].min()), float(n_atoms[m].max())],
                "n": int(m.sum()),
                "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                "mae": float(np.mean(abs_err[m])),
                "bias": float(np.mean(err[m])),
            }
        )
    group_rows: dict[str, list[float]] = {}
    for i, row in enumerate(rows):
        info = folds["rows"][str(int(row))]
        group_rows.setdefault(info["group_key_sha256"], []).append(float(abs_err[i]))
    worst = sorted(
        (
            {"group_key_sha256": k, "n": len(v), "mean_abs_error": float(np.mean(v))}
            for k, v in group_rows.items()
            if len(v) >= 5
        ),
        key=lambda x: -x["mean_abs_error"],
    )[:10]
    return {
        "fold": fold,
        "key": key,
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "bias": float(np.mean(err)),
        "correlations": correlations,
        "by_atom_count_quartile": by_size,
        "worst_scaffold_groups": worst,
    }


def _run_key(arm: str, recipe: str, fold: int, seed: int) -> str:
    direct = f"F{fold}_{arm}_{recipe}_s{seed}"
    if (C.RUNS / direct / "train.json").exists():
        return direct
    alias = f"F{fold}_{arm}_sel_s{seed}"
    if fold in (1, 2) and (C.RUNS / alias / "train.json").exists():
        return alias
    return direct


def run_diagnostics(seed: int = 0) -> dict[str, Any]:
    recipe_lock = C.read_json(C.RECIPE_LOCK)
    a_recipe = recipe_lock["selection"]["A"]
    c_recipe = recipe_lock["selection"]["C"]
    spec = C.load_frozen_spec()
    out = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "seed": int(seed),
        "monitor_used": False,
        "folds": [],
    }
    for fold in C.FOLDS:
        dev_cache, rows = _dev_cache_and_rows(fold)
        fold_view = D.build_fold_view(
            fold,
            cache=C.load_fit_cache()["cache"],
            y_fit=np.asarray(C.load_fit_labels()["y"], dtype=np.float64),
            folds=F.load_folds(),
        )
        key_a = _run_key("A", a_recipe, fold, seed)
        key_c = _run_key("C", c_recipe, fold, seed)
        key_e = f"F{fold}_E_R0_s{seed}"
        fold_out = {"fold": fold, "models": {}}
        for label, key in (("A", key_a), ("C", key_c), ("E", key_e)):
            model, payload = _load_model(key)
            batch = D.collate(dev_cache, list(range(dev_cache["n_molecules"])))
            base = _predict_batch(model, batch, fold_view.target_norm)
            perturb = {}
            settings = {
                "s2_joint_swap": (("path_cat", "path_end63"), True),
                "s2_independent_swap": (("path_cat", "path_end63"), False),
                "s2_full_path_swap": (("path_cat", "path_end63", "path_end", "path_closure"), True),
            }
            for name, (keys, same) in settings.items():
                pb = _permute_segments(batch, keys, same_perm=same, seed=12345 + fold * 10 + len(name))
                pred = _predict_batch(model, pb, fold_view.target_norm)
                perturb[name] = {
                    "rmse": D.rmse(pred, fold_view.y_dev),
                    "mae": D.mae(pred, fold_view.y_dev),
                    "mean_abs_delta": float(np.mean(np.abs(pred - base))),
                    "max_abs_delta": float(np.max(np.abs(pred - base))),
                }
            pb = _global_train_mean(batch, np.asarray(fold_view.train_norm["global_mean"], dtype=np.float32))
            pred = _predict_batch(model, pb, fold_view.target_norm)
            perturb["global_to_train_mean"] = {
                "rmse": D.rmse(pred, fold_view.y_dev),
                "mae": D.mae(pred, fold_view.y_dev),
                "mean_abs_delta": float(np.mean(np.abs(pred - base))),
                "max_abs_delta": float(np.max(np.abs(pred - base))),
            }
            fold_out["models"][label] = {
                "key": key,
                "recipe": payload["recipe"],
                "baseline": {
                    "rmse": D.rmse(base, fold_view.y_dev),
                    "mae": D.mae(base, fold_view.y_dev),
                },
                "perturbations": perturb,
                "s2_invariance_max_abs_delta": max(
                    v["max_abs_delta"] for k, v in perturb.items() if k.startswith("s2_")
                ),
            }
        fold_out["error_structure_A"] = _error_structure(fold, key_a, dev_cache)
        out["folds"].append(fold_out)
    # Aggregate model-level perturbation summaries.
    summary = {}
    for label in ("A", "C", "E"):
        rows = [f["models"][label] for f in out["folds"]]
        summary[label] = {
            "baseline_rmse_mean": float(np.mean([r["baseline"]["rmse"] for r in rows])),
            "s2_joint_swap_mean_abs_delta": float(np.mean([r["perturbations"]["s2_joint_swap"]["mean_abs_delta"] for r in rows])),
            "s2_independent_swap_mean_abs_delta": float(np.mean([r["perturbations"]["s2_independent_swap"]["mean_abs_delta"] for r in rows])),
            "s2_full_path_swap_mean_abs_delta": float(np.mean([r["perturbations"]["s2_full_path_swap"]["mean_abs_delta"] for r in rows])),
            "global_to_train_mean_mean_abs_delta": float(np.mean([r["perturbations"]["global_to_train_mean"]["mean_abs_delta"] for r in rows])),
            "s2_invariance_max_abs_delta": float(np.max([r["s2_invariance_max_abs_delta"] for r in rows])),
        }
    out["summary"] = summary
    C.write_json(C.ANALYSIS / "diagnostics_v2.json", out)
    return out


__all__ = ["run_diagnostics"]
