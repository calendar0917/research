"""Aggregate every recorded unit into the deliverable tables.

Produces
--------
``analysis/molecular_results.csv``      one row per (dataset, model, seed)
``analysis/tu_results.csv``             one row per (dataset, model, fold, seed)
``analysis/paired_differences.csv``     C1 - A0 paired differences (molecules by
                                        row, TU by outer fold) with bootstrap CIs
``analysis/sample_diagnostics.json``    per-sample improvement/regression counts,
                                        extreme-error concentration, top-k removal
                                        sensitivity, size-bin subgroups
``analysis/statistics.json``            seed dispersion, gap summaries, bootstrap
                                        protocol, failure counts
``analysis/literature_reference.csv``   fixed external references (no live lookup)

Every number in ``REPORT.md`` must be traceable to one of these files.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import ptb_common as C

import mb2_metrics as MET

from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    idx = rng.integers(0, values.size, size=(int(n_boot), values.size))
    samples = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.percentile(samples, 2.5)),
        "hi": float(np.percentile(samples, 97.5)),
        "n": int(values.size),
        "n_bootstrap": int(n_boot),
        "frac_positive": float((samples > 0).mean()),
    }


# ---------------------------------------------------------------------------
# molecule tables
# ---------------------------------------------------------------------------


def molecular_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_of = {d: C.MOLECULE_DATASETS[d]["task"] for d in C.MOLECULE_ORDER}
    for dataset in C.MOLECULE_ORDER:
        entry = manifest.get("datasets", {}).get(dataset)
        if not entry:
            rows.append(
                {
                    "dataset": dataset,
                    "status": "NOT_RUN_BUDGET",
                    "result_label": "NOT_RUN_BUDGET",
                }
            )
            continue
        by_unit = {u["unit"]: u for u in entry["units"]}
        for model in C.MODELS:
            for seed in C.MOLECULE_SEEDS:
                unit = C.mol_run_name(dataset, model, seed)
                rec = by_unit.get(unit)
                if rec is None:
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": int(seed),
                            "status": "NOT_RUN_BUDGET",
                            "result_label": "NOT_RUN_BUDGET",
                        }
                    )
                    continue
                run_dir = C.mol_run_dir(dataset, model, seed)
                test_eval_path = run_dir / "test_eval.json"
                test = C.read_json(test_eval_path) if test_eval_path.exists() else None
                row = {
                    "dataset": dataset,
                    "task": task_of[dataset],
                    "model": model,
                    "seed": int(seed),
                    "epochs_run": rec["epochs_run"],
                    "best_epoch": rec["best_epoch"],
                    "parameters": rec["parameters"],
                    "n_train": rec["n_train"],
                    "n_valid": rec["n_valid"],
                    "n_test": None if test is None else test["n_test"],
                    "gpu_hours": rec["gpu_hours"],
                    "checkpoint_sha256": rec["checkpoint_sha256"],
                    "budget_limited_possible": rec["budget_limited_possible"],
                    "status": "COMPLETE" if test is not None else "TRAINED_NOT_SCORED",
                    "result_label": C.RESULT_LABEL_MOL if test is not None else "NEW_INTERNAL_VALID_ONLY",
                }
                valid = rec["valid_metrics"]
                train = rec["train_metrics"]
                row["valid_primary"] = float(MET.primary_value(task_of[dataset], valid))
                if test is not None:
                    row["test_primary"] = float(MET.primary_value(task_of[dataset], test["metrics"]))
                else:
                    row["test_primary"] = None
                for k in ("rmse", "mae", "r2", "roc_auc", "pr_auc", "macro_roc_auc", "macro_pr_auc", "masked_bce", "bce"):
                    if k in valid:
                        row[f"valid_{k}"] = float(valid[k])
                    if test is not None and k in test["metrics"]:
                        row[f"test_{k}"] = float(test["metrics"][k])
                    if k in train:
                        row[f"train_{k}"] = float(train[k])
                if test is not None and test.get("official_ogb_roc_auc"):
                    row["test_official_ogb_roc_auc"] = float(test["official_ogb_roc_auc"]["roc_auc"])
                task = task_of[dataset]
                if row["test_primary"] is not None:
                    row["valid_minus_test_primary"] = float(row["valid_primary"] - row["test_primary"])
                if task == "regression":
                    row["train_minus_valid_rmse"] = float(train["rmse"] - valid["rmse"])
                elif task == "binary":
                    row["train_minus_valid_roc_auc"] = float(train["roc_auc"] - valid["roc_auc"])
                else:
                    row["train_minus_valid_macro_roc_auc"] = float(
                        train["macro_roc_auc"] - valid["macro_roc_auc"]
                    )
                rows.append(row)
    return rows


def _load_preds(run_dir: Path, split: str) -> np.ndarray | None:
    path = run_dir / f"{split}_predictions.npz"
    if not path.exists():
        return None
    return np.load(path)


def _primary_from_predictions(task: str, pred: np.ndarray, y: np.ndarray, target_norm) -> float:
    metrics = MET.task_metrics(task, pred, y, target_norm, [f"t{i}" for i in range(np.asarray(pred).reshape(len(pred), -1).shape[1])])
    return float(MET.primary_value(task, metrics))


def molecule_pair_stats(dataset: str, task: str, n_boot: int, rng: np.random.Generator) -> dict[str, Any]:
    """Paired C1 - A0 differences on identical Test rows, with row bootstrap."""
    import ptb_data as D

    view = D.molecule_view(dataset)
    out: dict[str, Any] = {"dataset": dataset, "task": task, "primary_metric": C.MOLECULE_DATASETS[dataset]["primary_metric"]}
    per_seed: list[dict[str, Any]] = []
    for seed in C.MOLECULE_SEEDS:
        a0 = _load_preds(C.mol_run_dir(dataset, C.MODEL_A0, seed), "test")
        c1 = _load_preds(C.mol_run_dir(dataset, C.MODEL_C1, seed), "test")
        if a0 is None or c1 is None:
            continue
        if not np.array_equal(a0["rows"], c1["rows"]):
            raise RuntimeError(f"{dataset}/s{seed}: A0 and C1 Test rows are not aligned")
        pa = np.asarray(a0["pred"], dtype=np.float64).reshape(len(a0["rows"]), -1)
        pc = np.asarray(c1["pred"], dtype=np.float64).reshape(len(c1["rows"]), -1)
        y = np.asarray(a0["y_true"], dtype=np.float64)
        if task == "regression":
            scale = float(view.target_norm["scale"])
            offset = float(view.target_norm["mean"])
            raw_a = pa[:, 0] * scale + offset
            raw_c = pc[:, 0] * scale + offset
            yv = y[:, 0]
            ea = np.abs(raw_a - yv)
            ec = np.abs(raw_c - yv)
            delta_rows = ec - ea  # >0 means C1 worse
            ma = _primary_from_predictions(task, pa, y, view.target_norm)
            mc = _primary_from_predictions(task, pc, y, view.target_norm)
            # paired bootstrap over molecular rows of the RMSE difference
            boot = []
            for _ in range(int(n_boot)):
                idx = rng.integers(0, yv.size, size=yv.size)
                ra = math.sqrt(float(np.mean((raw_a[idx] - yv[idx]) ** 2)))
                rc = math.sqrt(float(np.mean((raw_c[idx] - yv[idx]) ** 2)))
                boot.append(rc - ra)
            boot_arr = np.asarray(boot)
            per_seed.append(
                {
                    "seed": int(seed),
                    "A0_primary": ma,
                    "C1_primary": mc,
                    "delta_C1_minus_A0": mc - ma,
                    "paired_MAE_delta": float(ec.mean() - ea.mean()),
                    "frac_rows_C1_better": float((ec < ea).mean()),
                    "frac_rows_C1_worse": float((ec > ea).mean()),
                    "boot_lo": float(np.percentile(boot_arr, 2.5)),
                    "boot_hi": float(np.percentile(boot_arr, 97.5)),
                    "boot_frac_positive": float((boot_arr > 0).mean()),
                    "n_rows": int(yv.size),
                    "eval_unit": "molecule row",
                }
            )
        else:
            if task == "multitask":
                mask = np.isfinite(y)
                aucs_a, aucs_c = [], []
                for t in range(y.shape[1]):
                    m = mask[:, t]
                    if m.sum() == 0 or len(np.unique(y[m, t])) < 2:
                        continue
                    aucs_a.append(roc_auc_score(y[m, t], pa[m, t]))
                    aucs_c.append(roc_auc_score(y[m, t], pc[m, t]))
                ma, mc = float(np.mean(aucs_a)), float(np.mean(aucs_c))
            else:
                yv = y[:, 0]
                ma = float(roc_auc_score(yv, pa[:, 0]))
                mc = float(roc_auc_score(yv, pc[:, 0]))
            boot = []
            for _ in range(int(n_boot)):
                idx = rng.integers(0, y.shape[0], size=y.shape[0])
                if task == "multitask":
                    ys = y[idx]
                    m = np.isfinite(ys)
                    aa, cc = [], []
                    for t in range(ys.shape[1]):
                        mt = m[:, t]
                        if mt.sum() == 0 or len(np.unique(ys[mt, t])) < 2:
                            continue
                        aa.append(roc_auc_score(ys[mt, t], pa[idx][mt, t]))
                        cc.append(roc_auc_score(ys[mt, t], pc[idx][mt, t]))
                    boot.append(float(np.mean(cc) - np.mean(aa)))
                else:
                    ys = y[idx, 0]
                    if len(np.unique(ys)) < 2:
                        boot.append(0.0)
                    else:
                        boot.append(
                            float(
                                roc_auc_score(ys, pc[idx, 0]) - roc_auc_score(ys, pa[idx, 0])
                            )
                        )
            boot_arr = np.asarray(boot)
            per_seed.append(
                {
                    "seed": int(seed),
                    "A0_primary": ma,
                    "C1_primary": mc,
                    "delta_C1_minus_A0": mc - ma,
                    "boot_lo": float(np.percentile(boot_arr, 2.5)),
                    "boot_hi": float(np.percentile(boot_arr, 97.5)),
                    "boot_frac_positive": float((boot_arr > 0).mean()),
                    "n_rows": int(y.shape[0]),
                    "eval_unit": "molecule row",
                }
            )
    if per_seed:
        deltas = [s["delta_C1_minus_A0"] for s in per_seed]
        out["per_seed"] = per_seed
        out["delta_mean"] = float(np.mean(deltas))
        out["delta_std_over_seeds"] = float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0
        out["n_seeds"] = len(per_seed)
        out["direction"] = "higher_is_better" if C.MOLECULE_DATASETS[dataset]["direction"] == "higher" else "lower_is_better"
        out["ci_spans_zero"] = bool(min(s["boot_lo"] for s in per_seed) <= 0.0 <= max(s["boot_hi"] for s in per_seed))
    return out


# ---------------------------------------------------------------------------
# TU tables
# ---------------------------------------------------------------------------


def tu_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    scheme = protocol["tu_experiment"]["scheme"]
    rows: list[dict[str, Any]] = []
    for dataset in C.TU_DATASETS:
        for fold in range(int(scheme["n_splits"])):
            for seed in scheme["seeds"]:
                for model in C.MODELS:
                    run_dir = C.tu_run_dir(dataset, model, fold, int(seed))
                    run_json = run_dir / "run.json"
                    if not run_json.exists():
                        rows.append(
                            {
                                "dataset": dataset,
                                "fold": int(fold),
                                "seed": int(seed),
                                "model": model,
                                "status": "NOT_RUN_BUDGET",
                                "result_label": "NOT_RUN_BUDGET",
                            }
                        )
                        continue
                    rec = C.read_json(run_json)
                    rows.append(
                        {
                            "dataset": dataset,
                            "fold": int(fold),
                            "seed": int(seed),
                            "model": model,
                            "n_splits": int(rec["n_splits"]),
                            "n_train": int(rec["n_train"]),
                            "n_valid": int(rec["n_valid"]),
                            "n_test": int(rec["n_test"]),
                            "best_epoch": int(rec["checkpoint"]["best_epoch"]),
                            "valid_accuracy": float(rec["validation"]["accuracy"]),
                            "valid_macro_f1": float(rec["validation"]["macro_f1"]),
                            "valid_ce": float(rec["validation"]["cross_entropy"]),
                            "train_accuracy": float(rec["train"]["accuracy"]),
                            "train_ce": float(rec["train"]["cross_entropy"]),
                            "test_accuracy": None if rec["test"] is None else float(rec["test"]["accuracy"]),
                            "test_macro_f1": None if rec["test"] is None else float(rec["test"]["macro_f1"]),
                            "test_ce": None if rec["test"] is None else float(rec["test"]["cross_entropy"]),
                            "parameters": int(rec["parameters"]),
                            "n_node_types": int(rec["n_node_types"]),
                            "train_minus_valid_accuracy": float(
                                rec["train"]["accuracy"] - rec["validation"]["accuracy"]
                            ),
                            "gpu_hours": float(rec["timing"]["gpu_hours_estimate"]),
                            "peak_allocated_bytes": int(rec["timing"]["gpu_peak_allocated_bytes"]),
                            "checkpoint_sha256": rec["checkpoint"]["sha256"],
                            "partition_test_sha256": rec["partition"]["test"]["sha256"],
                            "status": "COMPLETE" if rec["test"] is not None else "TRAINED_NOT_SCORED",
                            "result_label": rec["result_label"],
                        }
                    )
    return rows


def tu_pair_stats(tu: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any], n_boot: int, rng) -> list[dict[str, Any]]:
    by_key = {(r["dataset"], r["fold"], r["seed"], r["model"]): r for r in tu if r.get("status") == "COMPLETE"}
    out: list[dict[str, Any]] = []
    scheme = protocol["tu_experiment"]["scheme"]
    for dataset in C.TU_DATASETS:
        deltas_val, deltas_test, matched = [], [], 0
        for fold in range(int(scheme["n_splits"])):
            for seed in scheme["seeds"]:
                a = by_key.get((dataset, fold, seed, C.MODEL_A0))
                c = by_key.get((dataset, fold, seed, C.MODEL_C1))
                if a is None or c is None:
                    continue
                if a["partition_test_sha256"] != c["partition_test_sha256"]:
                    raise RuntimeError(f"{dataset}/f{fold}/s{seed}: fold partitions differ between models")
                matched += 1
                deltas_val.append(c["valid_accuracy"] - a["valid_accuracy"])
                deltas_test.append(c["test_accuracy"] - a["test_accuracy"])
        if not deltas_test:
            continue
        d = np.asarray(deltas_test)
        boot = np.asarray([d[rng.integers(0, d.size, size=d.size)].mean() for _ in range(int(n_boot))])
        per_seed = {}
        for seed in scheme["seeds"]:
            vals = [
                (by_key[(dataset, f, seed, C.MODEL_C1)]["test_accuracy"] - by_key[(dataset, f, seed, C.MODEL_A0)]["test_accuracy"])
                for f in range(int(scheme["n_splits"]))
                if (dataset, f, seed, C.MODEL_C1) in by_key and (dataset, f, seed, C.MODEL_A0) in by_key
            ]
            if vals:
                per_seed[str(seed)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "n_folds": len(vals)}
        out.append(
            {
                "dataset": dataset,
                "kind": "tu_fold_paired",
                "primary_metric": "accuracy",
                "n_matched_fold_seed_pairs": matched,
                "delta_valid_mean": float(np.mean(deltas_val)),
                "delta_test_mean": float(np.mean(deltas_test)),
                "delta_test_std_across_fold_seed": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
                "delta_test_boot_lo": float(np.percentile(boot, 2.5)),
                "delta_test_boot_hi": float(np.percentile(boot, 97.5)),
                "delta_test_boot_frac_positive": float((boot > 0).mean()),
                "per_seed_delta": per_seed,
                "eval_unit": "outer fold (seeds reported separately)",
                "ci_spans_zero": bool(np.percentile(boot, 2.5) <= 0.0 <= np.percentile(boot, 97.5)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# sample diagnostics
# ---------------------------------------------------------------------------


def sample_diagnostics(manifest: Mapping[str, Any]) -> dict[str, Any]:
    import ptb_data as D

    out: dict[str, Any] = {}
    for dataset in C.MOLECULE_ORDER:
        entry = manifest.get("datasets", {}).get(dataset)
        if not entry or not entry.get("frozen"):
            continue
        view = D.molecule_view(dataset)
        test_cache, y_test, test_rows = D.molecule_test(dataset)
        atom_ptr = np.asarray(test_cache["atom_ptr"])
        n_atoms = (atom_ptr[1:] - atom_ptr[:-1]).astype(np.float64)
        per_seed = []
        for seed in C.MOLECULE_SEEDS:
            a0 = _load_preds(C.mol_run_dir(dataset, C.MODEL_A0, seed), "test")
            c1 = _load_preds(C.mol_run_dir(dataset, C.MODEL_C1, seed), "test")
            if a0 is None or c1 is None:
                continue
            pa = np.asarray(a0["pred"], dtype=np.float64).reshape(len(a0["rows"]), -1)
            pc = np.asarray(c1["pred"], dtype=np.float64).reshape(len(c1["rows"]), -1)
            y = np.asarray(a0["y_true"], dtype=np.float64)
            row: dict[str, Any] = {"seed": int(seed), "n_rows": int(y.shape[0])}
            if view.task == "regression":
                scale, offset = float(view.target_norm["scale"]), float(view.target_norm["mean"])
                ra, rc = pa[:, 0] * scale + offset, pc[:, 0] * scale + offset
                yv = y[:, 0]
                ea, ec = np.abs(ra - yv), np.abs(rc - yv)
                row.update(
                    {
                        "A0_mae": float(ea.mean()),
                        "C1_mae": float(ec.mean()),
                        "A0_rmse": float(np.sqrt(np.mean((ra - yv) ** 2))),
                        "C1_rmse": float(np.sqrt(np.mean((rc - yv) ** 2))),
                        "frac_rows_C1_better": float((ec < ea).mean()),
                        "frac_rows_C1_worse": float((ec > ea).mean()),
                        "frac_rows_tied": float((ec == ea).mean()),
                    }
                )
                for tag, err in (("A0", ea), ("C1", ec)):
                    k = max(1, int(math.ceil(0.05 * err.size)))
                    top = np.sort(err)[::-1][:k]
                    row[f"{tag}_top5pct_error_share"] = float(top.sum() / err.sum())
                    keep = np.argsort(err)[: err.size - k]
                    row[f"{tag}_rmse_drop_top5pct"] = float(np.sqrt(np.mean((err[keep]) ** 2)))
                # size bins
                bins = np.digitize(n_atoms, [15, 25])
                for b, name in enumerate(("small_le15", "mid_16_25", "large_gt25")):
                    m = bins == b
                    if m.sum() == 0:
                        continue
                    row[f"n_{name}"] = int(m.sum())
                    row[f"A0_mae_{name}"] = float(ea[m].mean())
                    row[f"C1_mae_{name}"] = float(ec[m].mean())
            else:
                if view.task == "binary":
                    yv = y[:, 0]
                    auc_a = float(roc_auc_score(yv, pa[:, 0]))
                    auc_c = float(roc_auc_score(yv, pc[:, 0]))
                    ap_a = float(average_precision_score(yv, pa[:, 0]))
                    ap_c = float(average_precision_score(yv, pc[:, 0]))
                else:
                    mask = np.isfinite(y)
                    aa, cc, apa, apc = [], [], [], []
                    for t in range(y.shape[1]):
                        m = mask[:, t]
                        if m.sum() == 0 or len(np.unique(y[m, t])) < 2:
                            continue
                        aa.append(roc_auc_score(y[m, t], pa[m, t]))
                        cc.append(roc_auc_score(y[m, t], pc[m, t]))
                        apa.append(average_precision_score(y[m, t], pa[m, t]))
                        apc.append(average_precision_score(y[m, t], pc[m, t]))
                    auc_a, auc_c = float(np.mean(aa)), float(np.mean(cc))
                    ap_a, ap_c = float(np.mean(apa)), float(np.mean(apc))
                b0 = np.digitize(n_atoms, [15, 25])
                row.update(
                    {
                        "A0_auroc": auc_a,
                        "C1_auroc": auc_c,
                        "A0_pr_auc": ap_a,
                        "C1_pr_auc": ap_c,
                        "n_positive": int(np.nansum(y[:, 0] > 0.5)),
                        "n_negative": int(np.nansum(y[:, 0] <= 0.5)),
                    }
                )
                for b, name in enumerate(("small_le15", "mid_16_25", "large_gt25")):
                    m = b0 == b
                    row[f"n_{name}"] = int(m.sum())
            per_seed.append(row)
        out[dataset] = {
            "task": view.task,
            "n_test": int(len(test_rows)),
            "per_seed": per_seed,
            "size_bin_definition": "n_atoms <=15 / 16-25 / >25 (descriptive only)",
        }
    return out


# ---------------------------------------------------------------------------
# literature references (fixed, no live lookup)
# ---------------------------------------------------------------------------


def literature_rows() -> list[dict[str, Any]]:
    refs = [
        {
            "dataset": "ZINC-12k",
            "model": "DCC-L (this repository, frozen 4 seeds)",
            "metric": "test_MAE",
            "value": 0.067319,
            "spread": 0.000709,
            "source": "tracks/structural-sufficiency/zinc_performance_search_v1 (frozen)",
            "grade": "FROZEN_HISTORICAL_OTHER_PROTOCOL",
            "parameters": "not recorded in this table",
            "input_information": "ZINC-12k atom/bond features of the frozen pipeline",
            "reproduced_here": False,
            "comparable_to_C1": False,
            "why_not_comparable": "different dataset, split, task and a ZINC-specific model family",
        },
        {
            "dataset": "ZINC-12k",
            "model": "GRIT (paper)",
            "metric": "test_MAE",
            "value": 0.059,
            "spread": 0.002,
            "source": "Ma et al. ICML 2023 (https://proceedings.mlr.press/v202/ma23c.html)",
            "grade": "LITERATURE_UNMATCHED_PROTOCOL",
            "parameters": "as reported by the authors (not re-checked here)",
            "input_information": "as reported by the authors",
            "reproduced_here": False,
            "comparable_to_C1": False,
            "why_not_comparable": "different model, hyper-parameter budget and split protocol",
        },
        {
            "dataset": "ZINC-12k",
            "model": "ESA+ (paper)",
            "metric": "test_MAE",
            "value": 0.052,
            "spread": 0.001,
            "source": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12141427/",
            "grade": "LITERATURE_UNMATCHED_PROTOCOL",
            "parameters": "as reported by the authors (not re-checked here)",
            "input_information": "as reported by the authors",
            "reproduced_here": False,
            "comparable_to_C1": False,
            "why_not_comparable": "different model family; ESA+ ZINC-FULL numbers must not be mixed into ZINC-12k",
        },
        {
            "dataset": "ogbg-molhiv",
            "model": "GIN + virtual node (official leaderboard)",
            "metric": "test_ROC_AUC",
            "value": 0.7707,
            "spread": 0.0149,
            "source": "https://ogb.stanford.edu/docs/leader_graphprop/",
            "grade": "LITERATURE_SAME_OFFICIAL_SPLIT",
            "parameters": "as reported by the OGB leaderboard (not re-checked here)",
            "input_information": "OGB graph features with a virtual node",
            "reproduced_here": False,
            "comparable_to_C1": True,
            "why_not_comparable": "same official scaffold split and evaluator, different architecture",
        },
        {
            "dataset": "ogbg-molhiv",
            "model": "PNA (official leaderboard)",
            "metric": "test_ROC_AUC",
            "value": 0.7905,
            "spread": 0.0132,
            "source": "https://ogb.stanford.edu/docs/leader_graphprop/",
            "grade": "LITERATURE_SAME_OFFICIAL_SPLIT",
            "parameters": "as reported by the OGB leaderboard (not re-checked here)",
            "input_information": "OGB graph features + degree-scaled aggregators",
            "reproduced_here": False,
            "comparable_to_C1": True,
            "why_not_comparable": "same official split, different architecture",
        },
        {
            "dataset": "ogbg-molhiv",
            "model": "GINE (official leaderboard)",
            "metric": "test_ROC_AUC",
            "value": 0.7921,
            "spread": 0.0128,
            "source": "https://ogb.stanford.edu/docs/leader_graphprop/",
            "grade": "LITERATURE_SAME_OFFICIAL_SPLIT",
            "parameters": "as reported by the OGB leaderboard (not re-checked here)",
            "input_information": "OGB graph features with edge features",
            "reproduced_here": False,
            "comparable_to_C1": True,
            "why_not_comparable": "same official split, different architecture",
        },
        {
            "dataset": "ogbg-molhiv",
            "model": "CIN (official leaderboard)",
            "metric": "test_ROC_AUC",
            "value": 0.8094,
            "spread": 0.0057,
            "source": "https://ogb.stanford.edu/docs/leader_graphprop/",
            "grade": "LITERATURE_SAME_OFFICIAL_SPLIT",
            "parameters": "as reported by the OGB leaderboard (not re-checked here)",
            "input_information": "OGB graph features, hierarchical substructure encoding",
            "reproduced_here": False,
            "comparable_to_C1": True,
            "why_not_comparable": "same official split, different architecture",
        },
        {
            "dataset": "ogbg-molhiv",
            "model": "HIG (official leaderboard)",
            "metric": "test_ROC_AUC",
            "value": 0.8403,
            "spread": 0.0021,
            "source": "https://ogb.stanford.edu/docs/leader_graphprop/",
            "grade": "LITERATURE_SAME_OFFICIAL_SPLIT",
            "parameters": "as reported by the OGB leaderboard (not re-checked here)",
            "input_information": "OGB graph features, multi-scale",
            "reproduced_here": False,
            "comparable_to_C1": True,
            "why_not_comparable": "same official split, different architecture",
        },
        {
            "dataset": "NCI1",
            "model": "GIN (common paper benchmark)",
            "metric": "accuracy",
            "value": 82.7,
            "spread": None,
            "source": "Xu et al. 2019 reported cross-validation benchmark (as fixed by the round instruction)",
            "grade": "LITERATURE_UNMATCHED_PROTOCOL",
            "parameters": "not fixed by this round",
            "input_information": "node labels, degree features",
            "reproduced_here": False,
            "comparable_to_C1": False,
            "why_not_comparable": "folds and model-selection protocol are not guaranteed identical",
        },
        {
            "dataset": "PROTEINS",
            "model": "GIN (common paper benchmark)",
            "metric": "accuracy",
            "value": 76.2,
            "spread": None,
            "source": "Xu et al. 2019 reported cross-validation benchmark (as fixed by the round instruction)",
            "grade": "LITERATURE_UNMATCHED_PROTOCOL",
            "parameters": "not fixed by this round",
            "input_information": "node labels, degree features",
            "reproduced_here": False,
            "comparable_to_C1": False,
            "why_not_comparable": "folds and model-selection protocol are not guaranteed identical",
        },
        {
            "dataset": "IMDB-BINARY",
            "model": "GIN (common paper benchmark)",
            "metric": "accuracy",
            "value": 75.1,
            "spread": None,
            "source": "Xu et al. 2019 reported cross-validation benchmark (as fixed by the round instruction)",
            "grade": "LITERATURE_UNMATCHED_PROTOCOL",
            "parameters": "not fixed by this round",
            "input_information": "degree features only",
            "reproduced_here": False,
            "comparable_to_C1": False,
            "why_not_comparable": "folds and model-selection protocol are not guaranteed identical",
        },
        {
            "dataset": "IMDB-MULTI",
            "model": "GIN (common paper benchmark)",
            "metric": "accuracy",
            "value": 52.3,
            "spread": None,
            "source": "Xu et al. 2019 reported cross-validation benchmark (as fixed by the round instruction)",
            "grade": "LITERATURE_UNMATCHED_PROTOCOL",
            "parameters": "not fixed by this round",
            "input_information": "degree features only",
            "reproduced_here": False,
            "comparable_to_C1": False,
            "why_not_comparable": "folds and model-selection protocol are not guaranteed identical",
        },
    ]
    # frozen historical same-split MolNet rows, copied read-only from v2
    lit = C.MOLNET_V2 / "analysis/literature_matrix.csv"
    if lit.exists():
        with open(lit, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                refs.append(
                    {
                        "dataset": row.get("dataset", ""),
                        "model": row.get("model", ""),
                        "metric": row.get("metric", ""),
                        "value": row.get("value", ""),
                        "spread": row.get("spread", ""),
                        "source": f"{C.track_rel(lit)}::{row.get('source', '')}",
                        "grade": row.get("grade", "LITERATURE_UNMATCHED_PROTOCOL"),
                        "parameters": row.get("parameters", ""),
                        "input_information": row.get("features", ""),
                        "reproduced_here": False,
                        "comparable_to_C1": False,
                        "why_not_comparable": row.get("notes", "carried over unchanged from the frozen v2 matrix"),
                    }
                )
    return refs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", default=str(C.CONFIGS / "protocol.json"))
    ap.add_argument("--n-boot", type=int, default=C.STATS_BOOTSTRAP_REPLICATES)
    args = ap.parse_args(argv)
    protocol = C.read_json(args.protocol)
    manifest_path = C.ANALYSIS / "checkpoint_manifest.json"
    manifest = C.read_json(manifest_path) if manifest_path.exists() else {"datasets": {}}
    rng = np.random.default_rng(C.STATS_BOOTSTRAP_SEED)

    mol_rows = molecular_rows(manifest)
    _write_csv(C.ANALYSIS / "molecular_results.csv", mol_rows)

    tu = tu_rows(protocol)
    _write_csv(C.ANALYSIS / "tu_results.csv", tu)

    paired: list[dict[str, Any]] = []
    mol_stats: dict[str, Any] = {}
    for dataset in C.MOLECULE_ORDER:
        entry = manifest.get("datasets", {}).get(dataset)
        if not entry or not entry.get("frozen"):
            continue
        stat = molecule_pair_stats(dataset, C.MOLECULE_DATASETS[dataset]["task"], args.n_boot, rng)
        mol_stats[dataset] = stat
        for s in stat.get("per_seed", []):
            paired.append(
                {
                    "kind": "molecule_row_paired",
                    "dataset": dataset,
                    "seed": s["seed"],
                    "metric": stat["primary_metric"],
                    "A0": s["A0_primary"],
                    "C1": s["C1_primary"],
                    "delta_C1_minus_A0": s["delta_C1_minus_A0"],
                    "boot_lo": s.get("boot_lo"),
                    "boot_hi": s.get("boot_hi"),
                    "eval_unit": "molecule row",
                    "result_label": C.RESULT_LABEL_MOL,
                }
            )
    tu_stats = tu_pair_stats(tu, protocol, args.n_boot, rng)
    for stat in tu_stats:
        paired.append(
            {
                "kind": "tu_fold_paired",
                "dataset": stat["dataset"],
                "seed": "all",
                "metric": stat["primary_metric"],
                "A0": None,
                "C1": None,
                "delta_C1_minus_A0": stat["delta_test_mean"],
                "boot_lo": stat["delta_test_boot_lo"],
                "boot_hi": stat["delta_test_boot_hi"],
                "eval_unit": "outer fold",
                "result_label": C.TU_RESULT_LABEL,
            }
        )
    _write_csv(C.ANALYSIS / "paired_differences.csv", paired)

    diag = sample_diagnostics(manifest)
    C.write_json(C.ANALYSIS / "sample_diagnostics.json", diag)

    _write_csv(C.ANALYSIS / "literature_reference.csv", literature_rows())

    # seed dispersion / failure counts
    failures = {
        "molecule_units_expected": len(C.MOLECULE_ORDER) * len(C.MODELS) * len(C.MOLECULE_SEEDS),
        "molecule_units_complete": int(sum(1 for r in mol_rows if r.get("status") == "COMPLETE")),
        "molecule_units_not_run": int(sum(1 for r in mol_rows if r.get("status") == "NOT_RUN_BUDGET")),
        "molecule_units_trained_not_scored": int(sum(1 for r in mol_rows if r.get("status") == "TRAINED_NOT_SCORED")),
        "tu_units_expected": int(C.read_json(args.protocol)["tu_experiment"]["n_units"]),
        "tu_units_complete": int(sum(1 for r in tu if r.get("status") == "COMPLETE")),
        "tu_units_not_run": int(sum(1 for r in tu if r.get("status") == "NOT_RUN_BUDGET")),
    }
    seed_dispersion: dict[str, Any] = {}
    for dataset in C.MOLECULE_ORDER:
        for model in C.MODELS:
            vals = [
                r["test_primary"]
                for r in mol_rows
                if r.get("dataset") == dataset and r.get("model") == model and r.get("test_primary") is not None
            ]
            if vals:
                mean, std = _mean_std(vals)
                seed_dispersion[f"{dataset}:{model}"] = {
                    "test_primary_mean": mean,
                    "test_primary_std_over_seeds": std,
                    "n_seeds": len(vals),
                    "individual": vals,
                }
    stats_payload = {
        "schema": "lcrc-ptb-statistics-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "bootstrap": {
            "seed": C.STATS_BOOTSTRAP_SEED,
            "replicates": args.n_boot,
            "same_resample_indices_for_both_models": True,
            "molecule_unit": "molecule Test row",
            "tu_unit": "outer fold",
            "seed_variation_reported_separately": True,
        },
        "molecule_paired": mol_stats,
        "tu_paired": tu_stats,
        "seed_dispersion_test_primary": seed_dispersion,
        "counts": failures,
        "notes": [
            "bootstrap CI is conditional on the trained models of this round",
            "fold x seed scores are not treated as independent samples",
            "a CI that spans zero is reported as inconclusive, not as equivalence",
        ],
    }
    C.write_json(C.ANALYSIS / "statistics.json", stats_payload)
    C.log(
        f"wrote molecular_results.csv ({len(mol_rows)} rows), tu_results.csv ({len(tu)} rows), "
        f"paired_differences.csv ({len(paired)}), statistics.json, sample_diagnostics.json, literature_reference.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
