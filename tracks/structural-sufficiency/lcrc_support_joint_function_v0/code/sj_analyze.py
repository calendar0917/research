"""Main table, paired scaffold-cluster bootstrap, function decomposition, contrasts.

Everything is read-only with respect to the trained models: the analysis loads
the frozen reused baseline (arm A) and this round's B/C checkpoints, recomputes
the joint function table over Omega, and writes the three analytical artefacts.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import sj_common as C
import sj_gates as G
import sj_model as M
import sj_support as S
import sj_train as T

METRIC = {"hiv": ("roc_auc", "pr_auc"), "lipo": ("rmse", "mae")}


# ---------------------------------------------------------------------------
# artefacts of one arm
# ---------------------------------------------------------------------------


def arm_paths(task_key: str, arm: str) -> dict[str, Path]:
    if arm == C.ARM_A:
        r = C.REUSE[task_key]
        return {
            "dir": r["run_dir"],
            "record": r["run_json"],
            "curve": r["curve"],
            "checkpoint": r["checkpoint"],
            "valid_predictions": r["valid_predictions"],
            "test_predictions": r["test_predictions"],
        }
    d = C.run_dir(task_key, arm)
    return {
        "dir": d,
        "record": d / "run.json",
        "curve": d / "curve.csv",
        "checkpoint": d / "checkpoint.pt",
        "valid_predictions": d / "valid_predictions.npz",
        "test_predictions": d / "test_predictions.npz",
    }


CURVE_ALIASES = {
    "valid_rmse_raw": "valid_rmse",
    "valid_mae_raw": "valid_mae",
    "valid_r2_raw": "valid_r2",
}


def read_curve(path: Path) -> list[dict[str, float]]:
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = []
    for row in rows:
        parsed: dict[str, float] = {}
        for key, value in row.items():
            if key == "is_best":
                parsed[key] = 1.0 if str(value).strip().lower() in ("true", "1") else 0.0
            else:
                parsed[key] = float(value)
        for old, new in CURVE_ALIASES.items():
            if old in parsed and new not in parsed:
                parsed[new] = parsed[old]
        out.append(parsed)
    return out


def load_checkpoint_model(task_key: str, arm: str, support: S.Support, view, device) -> torch.nn.Module:
    path = arm_paths(task_key, arm)["checkpoint"]
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model, _info = M.make_arm(task_key, view, support, arm, int(C.SEED), device)
    model.load_state_dict({k: v.to(device) for k, v in ck["state_dict"].items()}, strict=True)
    model.to(device)
    model.eval()
    return model


def reuse_baseline_record(task_key: str) -> dict[str, Any]:
    """(metrics, provenance) of arm A, read from the reused frozen records."""
    r = C.REUSE[task_key]
    variants = {k: v for k, v in (r.get("expected") or {}).items()}
    if task_key == "hiv":
        run = C.read_json(r["run_json"])
        tests = C.read_json(r["test_scores"])["scores"][f"mol_ogbg-molhiv_s0_{C.ARM_A}"]
        val = C.read_json(r["validation_metrics"])["units"][f"{C.ARM_A}_s0"]
        best = {
            "train": {"roc_auc": val["train_roc_auc"], "pr_auc": val["train_pr_auc"]},
            "valid": {"roc_auc": val["roc_auc"], "pr_auc": val["pr_auc"], "bce": val["bce"]},
            "test": {"roc_auc": tests["metrics"]["roc_auc"], "pr_auc": tests["metrics"]["pr_auc"],
                     "bce": tests["metrics"]["bce"]},
        }
        extra = {
            "parameters": int(run["model"]["parameters"]),
            "gpu_hours": float(run["timing"]["gpu_hours_estimate"]),
            "mean_epoch_seconds": float(run["timing"]["mean_epoch_seconds"]),
            "valid_seconds_per_split": float(run["timing"]["valid_seconds_total"] / run["timing"]["epochs_run"])
            if "epochs_run" in run["timing"] else None,
            "best_epoch": int(tests["best_epoch"]),
            "checkpoint_sha256": str(tests["checkpoint_sha256"]),
            "test_label": str(tests["result_label"]),
            "n_train_eval_available": False,
            "test_curve_available": False,
            "train_curve_available": False,
        }
        return {"metrics": best, **extra}
    run = C.read_json(r["run_json"])
    tests = C.read_json(r["test_scores"])
    best = {
        "train": dict(run["metrics"]["train"]),
        "valid": dict(run["metrics"]["valid"]),
        "test": {"rmse": tests["metrics"]["rmse"], "mae": tests["metrics"]["mae"]},
    }
    extra = {
        "parameters": int(run["model"]["parameters"]),
        "gpu_hours": float(run["timing"]["gpu_hours"]),
        "mean_epoch_seconds": float(run["timing"]["mean_epoch_seconds"]),
        "valid_seconds_per_split": float(run["timing"]["inference_seconds_per_valid_split"]),
        "best_epoch": int(run["checkpoint"]["best_epoch"]),
        "checkpoint_sha256": str(run["checkpoint"]["sha256"]),
        "test_label": str(tests["status"]),
        "n_train_eval_available": True,
        "test_curve_available": False,
        "train_curve_available": True,
    }
    return {"metrics": best, **extra}


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------


def main_table(task_key: str, support: S.Support, device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    view, _tr, _va, _te, spec = S.task_caches(task_key)
    y, test_cache = T.task_targets(task_key, view)
    task = str(C.TASKS[task_key]["task"])
    pk, sk = METRIC[task_key]
    names = T.task_names(view)
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    a = reuse_baseline_record(task_key)
    for arm in C.ARMS:
        paths = arm_paths(task_key, arm)
        curve = read_curve(paths["curve"])
        if arm == C.ARM_A:
            rec = None
            model = load_checkpoint_model(task_key, arm, support, view, device)
            best = a["metrics"]
            params = int(a["parameters"])
            gpu_hours = a["gpu_hours"]
            wall = None
            best_epoch = int(a["best_epoch"])
            source = "reused"
            mean_epoch_seconds = a["mean_epoch_seconds"]
        else:
            rec = C.read_json(paths["record"])
            model = load_checkpoint_model(task_key, arm, support, view, device)
            best = rec["metrics"]["best_checkpoint"]
            params = int(rec["parameters"])
            gpu_hours = float(rec["timing"]["gpu_hours"])
            wall = float(rec["timing"]["process_wall_seconds"])
            best_epoch = int(rec["checkpoint"]["best_epoch"])
            source = "this_round"
            mean_epoch_seconds = float(rec["timing"]["mean_epoch_seconds"])
        joint_params = M.joint_module_parameters(model)
        supported = int(M.train_supported_parameters(model, support))

        # inference cost, measured in this process (device of the analysis run)
        t0 = time.perf_counter()
        pred_valid = T.predict_raw(model, view.valid_cache, device)
        t_valid = time.perf_counter() - t0
        n_valid = int(view.valid_cache["n_molecules"])
        m_valid = C.MET.task_metrics(task, pred_valid, y["valid"], view.target_norm, names)
        t0 = time.perf_counter()
        pred_test = T.predict_raw(model, test_cache, device)
        t_test = time.perf_counter() - t0
        m_test = C.MET.task_metrics(task, pred_test, y["test"], view.target_norm, names)
        checks = {
            "valid_primary_recheck": float(m_valid[pk]),
            "test_primary_recheck": float(m_test[pk]),
            "valid_primary_recorded": float(best["valid"][pk]),
            "test_primary_recorded": float(best["test"][pk]),
        }
        checks["valid_replay_abs_diff"] = abs(checks["valid_primary_recheck"] - checks["valid_primary_recorded"])
        checks["test_replay_abs_diff"] = abs(checks["test_primary_recheck"] - checks["test_primary_recorded"])

        last20 = curve[-20:]
        row = {
            "task": task_key,
            "arm": arm,
            "source": source,
            "parameters_total": params,
            "parameters_joint_module": int(joint_params),
            "parameters_joint_with_train_support": supported,
            "best_epoch": best_epoch,
            "epochs_run": int(len(curve)),
            "train_primary_best": float(best["train"][pk]) if pk in best["train"] else None,
            "valid_primary_best": float(best["valid"][pk]),
            "test_primary_best": float(best["test"][pk]),
            "train_secondary_best": float(best["train"][sk]) if sk in best["train"] else None,
            "valid_secondary_best": float(best["valid"][sk]),
            "test_secondary_best": float(best["test"][sk]),
            "valid_primary_final": float(curve[-1][f"valid_{pk}"]),
            "valid_secondary_final": float(curve[-1][f"valid_{sk}"]),
            "train_primary_final": float(curve[-1].get(f"train_{pk}")) if f"train_{pk}" in curve[-1] else None,
            "test_primary_final": float(curve[-1].get(f"test_{pk}")) if f"test_{pk}" in curve[-1] else None,
            "last20_valid_primary": float(np.mean([r[f"valid_{pk}"] for r in last20])),
            "last20_valid_secondary": float(np.mean([r[f"valid_{sk}"] for r in last20])),
            "last20_train_primary": float(np.mean([r[f"train_{pk}"] for r in last20])) if f"train_{pk}" in last20[0] else None,
            "last20_test_primary": float(np.mean([r[f"test_{pk}"] for r in last20])) if f"test_{pk}" in last20[0] else None,
            "gpu_hours": gpu_hours,
            "mean_epoch_seconds": mean_epoch_seconds,
            "process_wall_seconds": wall,
            "inference_ms_per_molecule_valid": float(1000.0 * t_valid / n_valid),
            "inference_ms_per_molecule_test": float(1000.0 * t_test / int(test_cache["n_molecules"])),
            "inference_device": str(device),
            "valid_primary_replay_abs_diff": float(checks["valid_replay_abs_diff"]),
            "test_primary_replay_abs_diff": float(checks["test_replay_abs_diff"]),
        }
        rows.append(row)
        detail[arm] = {
            "metrics": {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()}
                        for k, v in best.items()},
            "replay": checks,
            "curve_epochs": len(curve),
            "record": C.track_rel(paths["record"]),
            "checkpoint": C.track_rel(paths["checkpoint"]),
            "checkpoint_sha256": C.sha256_file(paths["checkpoint"]),
            "valid_predictions": C.track_rel(paths["valid_predictions"]),
            "test_predictions": C.track_rel(paths["test_predictions"]),
            "table_column_notes": {
                "parameters_total": "all trainable parameters of the model",
                "parameters_joint_module": "parameters of the replaced S2 joint channel only",
                "parameters_joint_with_train_support": (
                    "joint-channel parameters actually reachable by a Train structure "
                    "(A: used embedding rows + projection; B: Omega rows; C: rank(X) rows)"
                ),
            },
            "train_supported_note": (
                "B has fewer total parameters than A here; the support-indexed table is smaller than "
                "path_u + path_v + joint_proj on both tasks"
            ),
        }
    return rows, detail


# ---------------------------------------------------------------------------
# scaffolds + paired cluster bootstrap
# ---------------------------------------------------------------------------


def murcko_scaffolds(smiles: Sequence[str]) -> list[str]:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    out: list[str] = []
    for text in smiles:
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            out.append("")
            continue
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))
        except Exception:
            out.append("")
    return out


def cluster_ids(smiles: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    groups = murcko_scaffolds(smiles)
    uniq = {g: i for i, g in enumerate(sorted(set(groups)))}
    ids = np.asarray([uniq[g] for g in groups], dtype=np.int64)
    sizes = np.bincount(ids)
    return ids, {
        "source": C.SCAFFOLD_SOURCE,
        "n_clusters": int(np.unique(ids).size),
        "n_molecules": int(ids.size),
        "largest_cluster": int(sizes.max()) if sizes.size else 0,
        "mean_cluster_size": float(sizes.mean()) if sizes.size else 0.0,
        "n_singleton_clusters": int((sizes == 1).sum()),
    }


def metric_value(name: str, pred: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

    if name == "roc_auc":
        return float(roc_auc_score(y, pred))
    if name == "pr_auc":
        return float(average_precision_score(y, pred))
    if name == "rmse":
        return float(math.sqrt(np.mean((pred - y) ** 2)))
    if name == "mae":
        return float(mean_absolute_error(y, pred))
    raise KeyError(name)


def paired_cluster_bootstrap(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    metrics: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    uniq = np.unique(clusters)
    idx_of = {int(c): np.flatnonzero(clusters == c) for c in uniq}
    point = {m: metric_value(m, pred_b, y) - metric_value(m, pred_a, y) for m in metrics}
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    skipped = 0
    for _ in range(int(replicates)):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        rows = np.concatenate([idx_of[int(uniq[i])] for i in pick])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            skipped += 1
            continue
        for m in metrics:
            draws[m].append(metric_value(m, pred_b[rows], yy) - metric_value(m, pred_a[rows], yy))
    out: dict[str, Any] = {"point": point, "n_replicates": int(replicates), "n_skipped": int(skipped)}
    for m in metrics:
        v = np.asarray(draws[m], dtype=np.float64)
        out[m] = {
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "ci95_low": float(np.percentile(v, 2.5)),
            "ci95_high": float(np.percentile(v, 97.5)),
            "fraction_same_sign_as_point": float(np.mean(np.sign(v) == np.sign(out["point"][m]))),
        }
    return out


def drop_extreme_sensitivity(
    pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray, metrics: Sequence[str], ks: Sequence[int]
) -> dict[str, Any]:
    """Symmetric sensitivity: drop the union of each model's top-k absolute errors."""
    out: dict[str, Any] = {}
    for k in ks:
        drop = np.union1d(np.argsort(-np.abs(pred_a - y))[:k], np.argsort(-np.abs(pred_b - y))[:k])
        keep = np.setdiff1d(np.arange(y.size), drop)
        entry = {"n_dropped": int(drop.size), "n_kept": int(keep.size)}
        for m in metrics:
            entry[f"delta_{m}"] = float(metric_value(m, pred_b[keep], y[keep]) - metric_value(m, pred_a[keep], y[keep]))
        out[f"drop_top{k}_union"] = entry
    return out


# ---------------------------------------------------------------------------
# function decomposition
# ---------------------------------------------------------------------------


def weighted_stats(support: S.Support, values: np.ndarray) -> dict[str, float]:
    """RMS over Omega under type-equal and occurrence weights (channel-averaged)."""
    v = np.asarray(values, dtype=np.float64)
    per_row = np.sqrt(np.mean(np.square(v), axis=1))
    w = support.omega_count.astype(np.float64)
    w = w / w.sum()
    return {
        "rms_type_equal": float(np.sqrt(np.mean(np.square(v)))),
        "rms_occurrence": float(np.sqrt(np.sum(w * np.square(per_row)))),
        "max_row_norm": float(per_row.max()) if per_row.size else 0.0,
    }


def decomposition(task_key: str, support: S.Support, models: Mapping[str, torch.nn.Module]) -> dict[str, Any]:
    w_joint = {arm: M.joint_readout_matrix(models[arm]) for arm in (C.ARM_A, C.ARM_B, C.ARM_C)}
    out: dict[str, Any] = {
        "task": task_key,
        "omega": int(support.n_omega),
        "rank_X": int(support.rank),
        "joint_subspace_dim": int(support.joint_dim),
        "weights": "type-equal primary (every Omega element weight 1); occurrence weights reported separately",
        "per_arm": {},
    }
    for arm in (C.ARM_A, C.ARM_B, C.ARM_C):
        table = M.extract_table(models[arm], support)
        jadd = M.additive_part(support, table)
        jint = M.nonadditive_part(support, table)
        w_occ = support.omega_count.astype(np.float64)
        w_occ = w_occ / w_occ.sum()
        closure = {
            "projector_closure_max_abs": float(np.max(np.abs(support.P @ jadd - jadd))),
            "identity_sum_max_abs": float(np.max(np.abs(jadd + jint - table))),
            "inner_product_type_equal": float(np.sum(jadd * jint)),
            "inner_product_occurrence": float(np.sum(w_occ * np.sum(jadd * jint, axis=1))),
            "relative_inner_product_type_equal": float(
                np.sum(jadd * jint) / (np.sqrt(np.sum(jadd**2) * np.sum(jint**2)) + 1e-30)
            ),
            "cosine_type_equal": float(
                np.sum(jadd * jint) / (np.sqrt(np.sum(jadd**2) * np.sum(jint**2)) + 1e-30)
            ),
        }
        add = weighted_stats(support, jadd)
        inter = weighted_stats(support, jint)
        allj = weighted_stats(support, table)
        proj_add = jadd @ w_joint[arm].T   # [K, hidden]
        proj_int = jint @ w_joint[arm].T
        wj_add = weighted_stats(support, proj_add)
        wj_int = weighted_stats(support, proj_int)
        out["per_arm"][arm] = {
            "rms_all": allj,
            "rms_additive": add,
            "rms_interaction": inter,
            "relative_interaction_type_equal": float(inter["rms_type_equal"] / add["rms_type_equal"])
            if add["rms_type_equal"] else None,
            "relative_interaction_occurrence": float(inter["rms_occurrence"] / add["rms_occurrence"])
            if add["rms_occurrence"] else None,
            "closure": closure,
            "w_joint_shape": list(w_joint[arm].shape),
            "w_joint_rms_additive": wj_add,
            "w_joint_rms_interaction": wj_int,
            "w_joint_relative_interaction_type_equal": float(wj_int["rms_type_equal"] / wj_add["rms_type_equal"])
            if wj_add["rms_type_equal"] else None,
            "w_joint_relative_interaction_occurrence": float(wj_int["rms_occurrence"] / wj_add["rms_occurrence"])
            if wj_add["rms_occurrence"] else None,
        }
    out["coverage_of_joint_computation"] = {
        split: {
            "occurrence_coverage": support.coverage[split]["occurrence_coverage"],
            "molecule_fraction_with_any_seen_pair": support.coverage[split]["molecule_fraction_with_any_seen_pair"],
            "molecule_fraction_with_any_unseen_pair": support.coverage[split]["molecule_fraction_with_any_unseen_pair"],
            "molecule_fraction_without_s2": support.coverage[split]["molecule_fraction_without_s2"],
            "n_omega_rows_with_any_occurrence": support.coverage[split]["n_omega_rows_with_any_occurrence"],
        }
        for split in support.coverage
    }
    out["note"] = (
        "latent coordinates of different models are not aligned; magnitudes are compared within one model only. "
        "Under type-equal weights J_add and J_int are orthogonal by construction (P is an orthogonal projector); "
        "other weightings do not guarantee orthogonality."
    )
    return out


# ---------------------------------------------------------------------------
# frozen joint-function intervention (arm B)
# ---------------------------------------------------------------------------


def intervention(task_key: str, support: S.Support, model_b, device) -> dict[str, Any]:
    view, _tr, _va, _te, spec = S.task_caches(task_key)
    y, test_cache = T.task_targets(task_key, view)
    task = str(C.TASKS[task_key]["task"])
    pk, sk = METRIC[task_key]
    names = T.task_names(view)
    local = model_b.local
    with torch.no_grad():
        base_map = local.complete_code_map()
        table = M.extract_table(model_b, support)
        jadd = M.additive_part(support, table)
        jint = M.nonadditive_part(support, table)
        seen = torch.as_tensor(support.index_table >= 0, dtype=torch.bool, device=base_map.device)
        seen_rows = torch.as_tensor(np.nonzero(support.index_table >= 0)[0], dtype=torch.long, device=base_map.device)
        maps: dict[str, torch.Tensor] = {}
        maps["ORIGINAL"] = base_map.clone()
        for name, alpha in (("ADD_ONLY", 0.0), ("HALF_INTERACTION", 0.5)):
            m = base_map.clone()
            j = torch.as_tensor(jadd + alpha * jint, dtype=torch.float32, device=m.device)
            m[seen_rows] = j
            maps[name] = m
        del seen_rows
        del seen
    out: dict[str, Any] = {
        "task": task_key,
        "checkpoint": C.track_rel(arm_paths(task_key, C.ARM_B)["checkpoint"]),
        "modes": {
            "ORIGINAL": "J_add + J_int (the trained table itself)",
            "ADD_ONLY": "J_add = P J",
            "HALF_INTERACTION": "J_add + 0.5 J_int",
        },
        "unseen_policy": "unseen codes keep the trained table's fixed fallback value in every mode",
        "results": {},
    }
    base_logits: dict[str, np.ndarray] = {}
    for mode, table_map in maps.items():
        local.set_override(table_map)
        try:
            for split, cache in (("valid", view.valid_cache), ("test", test_cache)):
                pred = T.predict_raw(model_b, cache, device)
                m = C.MET.task_metrics(task, pred, y[split], view.target_norm, names)
                if mode == "ORIGINAL":
                    base_logits[split] = pred
                ref = base_logits[split]
                entry = out["results"].setdefault(split, {})[mode] = {
                    pk: float(m[pk]),
                    sk: float(m[sk]),
                    "delta_primary_vs_original": float(m[pk] - C.MET.task_metrics(task, ref, y[split], view.target_norm, names)[pk]),
                    "logit_mean_abs_change": float(np.mean(np.abs(pred - ref))),
                    "logit_max_abs_change": float(np.max(np.abs(pred - ref))),
                }
        finally:
            local.clear_override()
    # breakdown by the molecule's own joint-structure coverage
    cov = support.molecule_coverage
    buckets = [
        ("full_1.0", lambda f: np.isfinite(f) & (f >= 1.0 - 1e-12)),
        ("0.9_to_1.0", lambda f: np.isfinite(f) & (f >= 0.9) & (f < 1.0 - 1e-12)),
        ("0.5_to_0.9", lambda f: np.isfinite(f) & (f >= 0.5) & (f < 0.9)),
        ("below_0.5", lambda f: np.isfinite(f) & (f < 0.5)),
        ("no_s2", lambda f: ~np.isfinite(f)),
    ]
    breakdown: dict[str, Any] = {}
    for split in ("valid", "test"):
        f = np.asarray(cov.get(split, np.zeros(0, dtype=np.float32)), dtype=np.float64)
        if f.size == 0:
            continue
        base = base_logits[split]
        add = None
        local.set_override(maps["ADD_ONLY"])
        try:
            add = T.predict_raw(model_b, view.valid_cache if split == "valid" else test_cache, device)
        finally:
            local.clear_override()
        entry: dict[str, Any] = {}
        for name, fn in buckets:
            mask = fn(f)
            if not mask.any():
                continue
            item = {
                "n_molecules": int(mask.sum()),
                "mean_logit_change_add_only": float(np.mean(np.abs(add[mask] - base[mask]))),
            }
            if len(np.unique(y[split][mask])) > 1 and pk in ("roc_auc", "pr_auc"):
                item[f"{pk}_original"] = float(metric_value(pk, base[mask], y[split][mask]))
                item[f"{pk}_add_only"] = float(metric_value(pk, add[mask], y[split][mask]))
            elif split and task == "regression":
                item["rmse_original"] = float(metric_value("rmse", base[mask], y[split][mask]))
                item["rmse_add_only"] = float(metric_value("rmse", add[mask], y[split][mask]))
            entry[name] = item
        breakdown[split] = entry
    out["coverage_breakdown"] = breakdown
    out["interpretation_guard"] = (
        "this intervention only asks whether the trained model uses the non-additive part of its own joint "
        "function; it is not a generalisation test and cannot attribute the trained B-vs-C difference to it"
    )
    return out


# ---------------------------------------------------------------------------
# interaction contrasts over real structure combinations
# ---------------------------------------------------------------------------


def atom_names(spec) -> list[str]:
    return list(spec.atom_type_names)


def bond_names(spec) -> list[str]:
    return list(spec.bond_type_names)


def describe(pair: Mapping[str, Any], spec) -> dict[str, Any]:
    atoms = atom_names(spec)
    bonds = bond_names(spec)
    m = int(pair["middle_atom_type"])
    a = int(pair["end_atom_type"])
    b1 = int(pair["bond_1"]) - 1
    b2 = int(pair["bond_2"]) - 1
    return {
        "readable": (
            f"root(-[{bonds[b1]}]-)-> {atoms[m]} -[{bonds[b2]}]- {atoms[a]}"
            if 0 <= m < len(atoms) and 0 <= a < len(atoms) and 0 <= b1 < len(bonds) and 0 <= b2 < len(bonds)
            else "undecodable"
        ),
        "middle_atom": atoms[m] if 0 <= m < len(atoms) else f"code {m}",
        "end_atom": atoms[a] if 0 <= a < len(atoms) else f"code {a}",
        "bond_1_root_to_middle": bonds[b1] if 0 <= b1 < len(bonds) else f"code {b1}",
        "bond_2_middle_to_end": bonds[b2] if 0 <= b2 < len(bonds) else f"code {b2}",
        "root_atom_type": "not part of the joint function's argument",
    }


def interaction_contrasts(task_key: str, support: S.Support, models, spec) -> dict[str, Any]:
    rect = G.rectangles(support)
    w = {arm: M.joint_readout_matrix(models[arm]) for arm in (C.ARM_A, C.ARM_B)}
    tables = {arm: M.extract_table(models[arm], support) for arm in (C.ARM_A, C.ARM_B)}
    docfreq = support.omega_docfreq
    counts = support.omega_count
    support_min = docfreq[rect].min(axis=1) if rect.size else np.zeros(0, dtype=np.int64)
    out: dict[str, Any] = {
        "task": task_key,
        "n_rectangles_in_omega": int(rect.shape[0]),
        "definition": (
            "D = J(p1,e1) + J(p2,e2) - J(p1,e2) - J(p2,e1) for quadruples whose four pairs all occur in "
            "Omega; it cancels every purely path-specific and every purely endpoint-specific term"
        ),
        "candidate_rule": {
            "min_pair_docfreq": int(C.CONTRAST_MIN_PAIR_DOCFREQ),
            "max_candidates": int(C.CONTRAST_MAX_CANDIDATES),
            "ranking": "descending || W_joint D ||_2 of arm B's best-Valid checkpoint",
            "support_definition": "minimum distinct-Train-molecule support over the four pairs",
        },
        "per_arm": {},
    }
    for arm in (C.ARM_A, C.ARM_B):
        d = tables[arm][rect[:, 3]] - tables[arm][rect[:, 2]] - tables[arm][rect[:, 1]] + tables[arm][rect[:, 0]]
        proj = d @ w[arm].T
        out["per_arm"][arm] = {
            "n_rectangles": int(rect.shape[0]),
            "rms_contrast": float(np.sqrt(np.mean(np.square(d)))) if rect.size else 0.0,
            "max_abs_contrast": float(np.max(np.abs(d))) if rect.size else 0.0,
            "rms_w_joint_contrast": float(np.sqrt(np.mean(np.square(proj)))) if rect.size else 0.0,
            "max_w_joint_norm": float(np.max(np.linalg.norm(proj, axis=1))) if rect.size else 0.0,
        }
    if not rect.size:
        out["candidates"] = []
        out["note"] = "Omega contains no rectangle, so no interaction contrast can be computed"
        out["passed_no_fabrication"] = True
        return out
    proj_b = (
        tables[C.ARM_B][rect[:, 3]]
        - tables[C.ARM_B][rect[:, 2]]
        - tables[C.ARM_B][rect[:, 1]]
        + tables[C.ARM_B][rect[:, 0]]
    ) @ w[C.ARM_B].T
    norms = np.linalg.norm(proj_b, axis=1)
    eligible = np.flatnonzero(support_min >= int(C.CONTRAST_MIN_PAIR_DOCFREQ))
    order = eligible[np.argsort(-norms[eligible])][: int(C.CONTRAST_MAX_CANDIDATES)]
    # example molecules for the selected pairs
    view, train_cache, valid_cache, test_cache, spec2 = S.task_caches(task_key)
    n_nbr = int(support.n_nbr_codes)
    caches = {"train": train_cache, "valid": valid_cache, "test": test_cache}
    recs = {split: S.split_records(cache) for split, cache in caches.items()}
    smiles = {
        split: [str(cache["smiles"][i]) for i in range(int(cache["n_molecules"]))]
        for split, cache in caches.items()
    }
    candidates = []
    for rank, ridx in enumerate(order.tolist(), start=1):
        quads = rect[ridx]
        pairs = [int(support.omega_code[i]) for i in quads]
        pd = [int(docfreq[i]) for i in quads]
        pc = [int(counts[i]) for i in quads]
        p1, p2 = pairs[0] // n_nbr, pairs[2] // n_nbr
        e1, e2 = pairs[0] % n_nbr, pairs[1] % n_nbr
        d_vec = tables[C.ARM_B][quads[3]] - tables[C.ARM_B][quads[2]] - tables[C.ARM_B][quads[1]] + tables[C.ARM_B][quads[0]]
        s_vec = w[C.ARM_B] @ d_vec
        splitting = {
            f"p({p1})": int(sum(support.omega_p == p1)),
            f"p({p2})": int(sum(support.omega_p == p2)),
            f"e({e1})": int(sum(support.omega_e == e1)),
            f"e({e2})": int(sum(support.omega_e == e2)),
        }
        entry = {
            "rank": rank,
            "rect_index": int(ridx),
            "p1": int(p1),
            "p2": int(p2),
            "e1": int(e1),
            "e2": int(e2),
            "pairs": {
                "(p1,e1)": {"code": int(pairs[0]), "docfreq_train": int(pd[0]), "occurrences_train": int(pc[0])},
                "(p1,e2)": {"code": int(pairs[1]), "docfreq_train": int(pd[1]), "occurrences_train": int(pc[1])},
                "(p2,e1)": {"code": int(pairs[2]), "docfreq_train": int(pd[2]), "occurrences_train": int(pc[2])},
                "(p2,e2)": {"code": int(pairs[3]), "docfreq_train": int(pd[3]), "occurrences_train": int(pc[3])},
            },
            "structures": {
                "(p1,e1)": describe(support.decode(p1, e1), spec),
                "(p1,e2)": describe(support.decode(p1, e2), spec),
                "(p2,e1)": describe(support.decode(p2, e1), spec),
                "(p2,e2)": describe(support.decode(p2, e2), spec),
            },
            "support": {
                "min_pair_docfreq_train": int(support_min[ridx]),
                "pair_docfreq_train": pd,
                "pair_occurrences_train": pc,
                "paths_sharing_p_or_e": splitting,
            },
            "contrast": {
                "norm": float(np.linalg.norm(d_vec)),
                "w_joint_norm": float(np.linalg.norm(s_vec)),
                "rms_entries": float(np.sqrt(np.mean(np.square(d_vec)))),
                "w_joint_rms_entries": float(np.sqrt(np.mean(np.square(s_vec)))),
                "largest_abs_entry": float(np.max(np.abs(d_vec))),
            },
            "split_support": {},
            "example_molecules": {},
            "counterexamples": {},
        }
        for split, rec in recs.items():
            code_split = S.pair_code(rec["path_cat"], rec["path_end63"], n_nbr)
            entry["split_support"][split] = {
                "occurrences_per_pair": [int((code_split == c).sum()) for c in pairs],
                "n_molecules_per_pair": [int(np.unique(rec["mol_of_path"][code_split == c]).size) for c in pairs],
            }
            mols = []
            for c in pairs:
                hit = np.unique(rec["mol_of_path"][code_split == c])
                for m in hit[:3].tolist():
                    mols.append({"pair_code": int(c), "row": int(np.asarray(caches[split]["rows"])[m]), "smiles": smiles[split][m]})
            entry["example_molecules"][split] = mols[:12]
        # counterexamples: the same p1/p2 or e1/e2 with the missing partners
        missing = []
        for p in (p1, p2):
            for e in (e1, e2):
                if int(p * n_nbr + e) not in set(pairs):
                    missing.append({"p": int(p), "e": int(e), "in_omega": False})
        entry["counterexamples"] = {"pairs_of_the_same_types_not_in_omega": missing}
        candidates.append(entry)
    out["candidates"] = candidates
    out["n_eligible_rectangles"] = int(eligible.size)
    out["guard"] = (
        "these are model-internal joint computation regularities, not causal interactions of molecular "
        "properties and not final prediction contributions"
    )
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_task(task_key: str, device) -> dict[str, Any]:
    view, _tr, _va, _te, spec = S.task_caches(task_key)
    support = S.build_task_support(task_key)
    models: dict[str, torch.nn.Module] = {}
    for arm in C.ARMS:
        path = arm_paths(task_key, arm)["checkpoint"]
        if not path.exists():
            raise FileNotFoundError(f"missing checkpoint {path}")
        models[arm] = load_checkpoint_model(task_key, arm, support, view, device)
    rows, detail = main_table(task_key, support, device)

    # paired scaffold-cluster bootstrap on Valid and Test
    y, test_cache = T.task_targets(task_key, view)
    preds = {arm: {} for arm in C.ARMS}
    for arm in C.ARMS:
        preds[arm]["valid"] = T.predict_raw(models[arm], view.valid_cache, device)[:, 0]
        preds[arm]["test"] = T.predict_raw(models[arm], test_cache, device)[:, 0]
    boot: dict[str, Any] = {}
    for split, cache in (("valid", view.valid_cache), ("test", test_cache)):
        ids, meta = cluster_ids([str(s) for s in cache["smiles"]])
        boot[split] = {"scaffolds": meta, "contrasts": {}}
        for a, b in ((C.ARM_A, C.ARM_B), (C.ARM_C, C.ARM_B), (C.ARM_A, C.ARM_C)):
            key = f"{b}-{a}"
            boot[split]["contrasts"][key] = paired_cluster_bootstrap(
                preds[a][split],
                preds[b][split],
                y[split],
                ids,
                METRIC[task_key],
                replicates=C.BOOTSTRAP_REPLICATES,
                seed=C.BOOTSTRAP_SEED,
            )
        if C.TASKS[task_key]["task"] == "regression":
            boot[split]["drop_extreme_sensitivity"] = {
                f"{C.ARM_B}-{C.ARM_A}": drop_extreme_sensitivity(
                    preds[C.ARM_A][split], preds[C.ARM_B][split], y[split], METRIC[task_key], (1, 5, 10)
                ),
                f"{C.ARM_C}-{C.ARM_A}": drop_extreme_sensitivity(
                    preds[C.ARM_A][split], preds[C.ARM_C][split], y[split], METRIC[task_key], (1, 5, 10)
                ),
                f"{C.ARM_B}-{C.ARM_C}": drop_extreme_sensitivity(
                    preds[C.ARM_C][split], preds[C.ARM_B][split], y[split], METRIC[task_key], (1, 5, 10)
                ),
            }
    dec = decomposition(task_key, support, models)
    inter = intervention(task_key, support, models[C.ARM_B], device)
    contrasts = interaction_contrasts(task_key, support, models, spec)
    # post-training re-check of the two parameterisation identities on the trained C
    post = {
        "gate10_trained_C": G.gate10_additive_constraint(task_key, support, models[C.ARM_C]),
        "gate11_trained_C": G.gate11_interaction_zero(task_key, support, models[C.ARM_C]),
    }
    return {
        "rows": rows,
        "detail": detail,
        "bootstrap": boot,
        "decomposition": dec,
        "intervention": inter,
        "contrasts": contrasts,
        "post_gates": post,
        "predictions": {arm: {split: preds[arm][split].tolist() for split in ("valid", "test")} for arm in C.ARMS},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Analysis of the joint-function round")
    ap.add_argument("--tasks", nargs="+", default=list(C.TASK_ORDER))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-predictions-export", action="store_true")
    args = ap.parse_args(argv)
    device = torch.device(args.device)
    payload: dict[str, Any] = {
        "schema": "lcrc-support-joint-function-analysis-v1",
        "round_id": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "created_utc": C.utcnow(),
        "env": C.env_info(),
        "tasks": {},
    }
    main_rows: list[dict[str, Any]] = []
    for task_key in args.tasks:
        C.log(f"analysis: {task_key}")
        res = run_task(task_key, device)
        payload["tasks"][task_key] = {k: v for k, v in res.items() if k != "rows"}
        payload["tasks"][task_key]["rows"] = res["rows"]
        main_rows.extend(res["rows"])
    fields: list[str] = []
    for row in main_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    C.write_csv(C.ANALYSIS / "main_table.csv", main_rows, fieldnames=fields)
    C.write_json(
        C.ANALYSIS / "main_table.json",
        {
            "schema": "lcrc-support-joint-function-main-table-v1",
            "round_id": C.ROUND_ID,
            "created_utc": C.utcnow(),
            "reference_delta": C.REFERENCE_DELTA,
            "rows": main_rows,
            "detail": {k: v["detail"] for k, v in payload["tasks"].items()},
            "bootstrap": {k: v["bootstrap"] for k, v in payload["tasks"].items()},
        },
    )
    C.write_json(
        C.ANALYSIS / "function_decomposition.json",
        {
            "schema": "lcrc-support-joint-function-decomposition-v1",
            "round_id": C.ROUND_ID,
            "created_utc": C.utcnow(),
            "tasks": {k: v["decomposition"] for k, v in payload["tasks"].items()},
            "intervention": {k: v["intervention"] for k, v in payload["tasks"].items()},
            "post_training_identity_checks": {k: v["post_gates"] for k, v in payload["tasks"].items()},
        },
    )
    C.write_json(
        C.ANALYSIS / "interaction_contrasts.json",
        {
            "schema": "lcrc-support-joint-function-interaction-contrasts-v1",
            "round_id": C.ROUND_ID,
            "created_utc": C.utcnow(),
            "tasks": {k: v["contrasts"] for k, v in payload["tasks"].items()},
        },
    )
    C.log("analysis artefacts written")
    print(
        C.stable_json(
            {
                task: {
                    "rows": [
                        {k: r[k] for k in ("arm", "best_epoch", "valid_primary_best", "test_primary_best", "parameters_total")}
                        for r in payload["tasks"][task]["rows"]
                    ]
                }
                for task in args.tasks
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
