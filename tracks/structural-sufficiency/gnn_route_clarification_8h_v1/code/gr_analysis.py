"""Prediction tables, paired contrasts, oracle-calibrated explanation error.

Everything here is a *fixed-state* analysis of frozen checkpoints: no training,
no selection, no seed search.  Extra splits (Val/Test) are observed and
reported, never used to pick a model; the only selection rules live in
``gr_select`` and are applied to Val RMSE.

Outputs live under ``analysis/tables/`` as JSON (machine readable) and CSV
(one row per metric) so the report can quote exact numbers.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import gr_common as C
import gr_data as D
import gr_select as S

BOOT_N = 2000
BOOT_SEED = C.BOOTSTRAP_SEED


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def table_dir():
    path = C.ANALYSIS_DIR / "tables"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_table(name: str, payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]] | None = None) -> None:
    C.write_json(table_dir() / f"{name}.json", payload)
    if rows is not None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        path = table_dir() / f"{name}.csv"
        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_value(row.get(k)) for k in keys})
        tmp.replace(path)


def _csv_value(value):
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def load_catalog() -> list[dict[str, Any]]:
    with gzip.open(C.ANALYSIS_DIR / "pair_catalog.jsonl.gz", "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# prediction tables
# ---------------------------------------------------------------------------


def load_predictions(stage: str, dataset: str, model: str, recipe: str) -> dict[str, dict[str, np.ndarray]]:
    path = C.task_dir(stage, dataset, model, recipe) / "best_predictions.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path)
    out = {}
    for split in C.SPLITS:
        out[split] = {
            "row_id": z[f"{split}_row_id"].astype(np.int64),
            "y_raw": z[f"{split}_y_raw"].astype(np.float64),
            "pred_raw": z[f"{split}_pred_raw"].astype(np.float64),
            "pred_std": z[f"{split}_pred_std"].astype(np.float64),
        }
    return out


def load_trajectory(stage: str, dataset: str, model: str, recipe: str) -> dict[str, Any]:
    path = C.task_dir(stage, dataset, model, recipe) / "trajectory.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return C.read_json(path)


def rmse(pred, y) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred, float) - np.asarray(y, float)) ** 2)))


def mae(pred, y) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(y, float))))


def _align(preds: Sequence[Mapping[str, np.ndarray]], split: str):
    common = sorted(set.intersection(*[set(p["row_id"].tolist()) for p in preds]))
    lookup = [{int(r): i for i, r in enumerate(p["row_id"].tolist())} for p in preds]
    idx = [np.asarray([lk[r] for r in common], dtype=np.int64) for lk in lookup]
    return np.asarray(common), [
        {"y_raw": p["y_raw"][i], "pred_raw": p["pred_raw"][i], "pred_std": p["pred_std"][i]} for p, i in zip(preds, idx)
    ]


def bootstrap_rmse_diff(pred_a, pred_b, y, *, n=BOOT_N, seed=BOOT_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    size = len(y)
    diffs = np.empty(n)
    ratios = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, size, size)
        ea = rmse(pred_a[idx], y[idx])
        eb = rmse(pred_b[idx], y[idx])
        diffs[i] = ea - eb
        ratios[i] = ea / eb
    return {
        "rmse_a": rmse(pred_a, y),
        "rmse_b": rmse(pred_b, y),
        "diff": float(diffs.mean()),
        "diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "ratio": float(ratios.mean()),
        "ratio_ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
        "prob_a_better": float(np.mean(diffs < 0)),
        "n_rows": int(size),
        "n_boot": n,
        "seed": seed,
    }


def stage1_table() -> dict[str, Any]:
    rows = []
    trajs: dict[tuple[str, str, str], dict[str, Any]] = {}
    preds: dict[tuple[str, str, str], dict[str, dict[str, np.ndarray]]] = {}
    for dataset in C.DATASETS:
        for arch in C.ARCHS:
            for recipe in C.RECIPE_ORDER:
                try:
                    traj = load_trajectory(C.STAGE1, dataset, arch, recipe)
                    pred = load_predictions(C.STAGE1, dataset, arch, recipe)
                except FileNotFoundError:
                    rows.append({"dataset": dataset, "model": arch, "recipe": recipe, "status": "MISSING"})
                    continue
                trajs[(dataset, arch, recipe)] = traj
                preds[(dataset, arch, recipe)] = pred
                rows.append(
                    {
                        "dataset": dataset,
                        "model": arch,
                        "recipe": recipe,
                        "status": "COMPLETE",
                        "selected_epoch": traj["selected"]["epoch"],
                        "valid_rmse_raw": traj["selected"]["valid_rmse_raw"],
                        "valid_mae_raw": traj["selected"]["valid_mae_raw"],
                        "test_rmse_raw": traj["selected"]["test_rmse_raw"],
                        "test_mae_raw": traj["selected"]["test_mae_raw"],
                        "train_rmse_raw": traj["selected"]["train_rmse_raw"],
                        "final_valid_rmse_raw": traj["final"]["valid_rmse_raw"],
                        "final_test_rmse_raw": traj["final"]["test_rmse_raw"],
                        "last20_valid_rmse_raw": traj["last20"]["valid_rmse_raw"],
                        "n_parameters": traj["n_parameters"],
                        "seconds_total": traj.get("seconds_total"),
                    }
                )
    if not trajs:
        return {"rows": rows, "status": "NO_RESULTS"}

    # Val-selected recipe per cell (frozen rule: minimum Val RMSE, ties R3).
    selected: dict[tuple[str, str], str] = {}
    for dataset in C.DATASETS:
        for arch in C.ARCHS:
            try:
                selected[(dataset, arch)] = S.selected_recipe(dataset, arch)
            except FileNotFoundError:
                pass

    contrasts = []
    for dataset in C.DATASETS:
        if (dataset, "L") not in selected:
            continue
        for arch in ("G0", "Gx"):
            if (dataset, arch) not in selected:
                continue
            for split in ("valid", "test"):
                _, blocks = _align(
                    [preds[(dataset, "L", selected[(dataset, "L")])][split], preds[(dataset, arch, selected[(dataset, arch)])][split]],
                    split,
                )
                stats = bootstrap_rmse_diff(blocks[0]["pred_raw"], blocks[1]["pred_raw"], blocks[0]["y_raw"])
                contrasts.append({"dataset": dataset, "split": split, "a": "L", "b": arch, **stats})
    # recipe interaction of the L-vs-GINE gap
    recipe_contrast = []
    for dataset in C.DATASETS:
        for arch in ("G0", "Gx"):
            for recipe in C.RECIPE_ORDER:
                keys = [(dataset, "L", recipe), (dataset, arch, recipe)]
                if not all(k in preds for k in keys):
                    continue
                _, blocks = _align([preds[keys[0]]["valid"], preds[keys[1]]["valid"]], "valid")
                gap = rmse(blocks[0]["pred_raw"], blocks[0]["y_raw"]) - rmse(blocks[1]["pred_raw"], blocks[1]["y_raw"])
                recipe_contrast.append({"dataset": dataset, "arch": arch, "recipe": recipe, "valid_gap_L_minus": gap})
    # Gx - G0 (input bridge) at the selected recipe
    bridge = []
    for dataset in C.DATASETS:
        if (dataset, "G0") not in selected or (dataset, "Gx") not in selected:
            continue
        for split in ("valid", "test"):
            _, blocks = _align(
                [preds[(dataset, "G0", selected[(dataset, "G0")])][split], preds[(dataset, "Gx", selected[(dataset, "Gx")])][split]],
                split,
            )
            stats = bootstrap_rmse_diff(blocks[1]["pred_raw"], blocks[0]["pred_raw"], blocks[0]["y_raw"])
            bridge.append({"dataset": dataset, "split": split, "a": "Gx", "b": "G0", **stats})
    payload = {
        "rows": rows,
        "selected_recipe": {f"{d}/{m}": r for (d, m), r in selected.items()},
        "selected_scores": {
            f"{d}/{m}": {
                "valid_rmse_raw": trajs[(d, m, r)]["selected"]["valid_rmse_raw"],
                "test_rmse_raw": trajs[(d, m, r)]["selected"]["test_rmse_raw"],
                "epoch": trajs[(d, m, r)]["selected"]["epoch"],
                "n_parameters": trajs[(d, m, r)]["n_parameters"],
            }
            for (d, m), r in selected.items()
        },
        "l_vs_gine": contrasts,
        "recipe_gap": recipe_contrast,
        "gx_vs_g0": bridge,
    }
    return payload


# ---------------------------------------------------------------------------
# paired MMP predictions (all 823 catalog pairs)
# ---------------------------------------------------------------------------


def _pair_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_pairs": 0}
    out = {"n_pairs": len(rows)}
    for key in ("abs_delta_error", "delta_error", "abs_dy"):
        out[f"{key}_mean"] = float(np.mean([r[key] for r in rows]))
    signs = [r for r in rows if abs(r["dy"]) >= 0.1 and r["sign_ok"] is not None]
    out["n_sign"] = len(signs)
    out["sign_accuracy"] = float(np.mean([r["sign_ok"] for r in signs])) if signs else None
    comp: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        comp[int(r["component"])].append(r["abs_delta_error"])
    comp_means = [float(np.mean(v)) for v in comp.values()]
    out["n_components"] = len(comp_means)
    out["component_equal_abs_delta_error"] = float(np.mean(comp_means)) if comp_means else None
    comp_sign: dict[int, list[float]] = defaultdict(list)
    for r in signs:
        comp_sign[int(r["component"])].append(1.0 if r["sign_ok"] else 0.0)
    out["component_equal_sign_accuracy"] = float(np.mean([np.mean(v) for v in comp_sign.values()])) if comp_sign else None
    out["support"] = "LOW_SUPPORT" if len(comp_means) < 30 else "OK"
    return out


def row_targets(b) -> dict[int, float]:
    out: dict[int, float] = {}
    for split in C.SPLITS:
        for i, row in enumerate(np.asarray(b.global_row_ids[split]).tolist()):
            out[int(row)] = float(b.targets_raw[split][i])
    return out


def mmp_predictions() -> dict[str, Any]:
    b = D.bundle("ogbg-mollipo")
    y = row_targets(b)
    catalog = load_catalog()
    models: dict[str, dict[str, np.ndarray]] = {}
    for name, stage, model, recipe in _model_specs():
        try:
            pred = load_predictions(stage, "ogbg-mollipo", model, recipe)
        except FileNotFoundError:
            continue
        row_map = {}
        for split in C.SPLITS:
            for row, value in zip(pred[split]["row_id"].tolist(), pred[split]["pred_raw"].tolist()):
                row_map[int(row)] = float(value)
        models[name] = {"row_map": row_map, "spec": {"stage": stage, "model": model, "recipe": recipe}}
    out: dict[str, Any] = {"models": {k: v["spec"] for k, v in models.items()}, "strata": {}, "per_model": {}, "pairs": {}}
    for name, block in models.items():
        rows = []
        for pair in catalog:
            a, bb = int(pair["row_a"]), int(pair["row_b"])
            if a not in block["row_map"] or bb not in block["row_map"]:
                continue
            dy = float(y[bb] - y[a])
            df = block["row_map"][bb] - block["row_map"][a]
            sign_ok = None if abs(dy) < 0.1 else bool(np.sign(df) == np.sign(dy))
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "row_a": a,
                    "row_b": bb,
                    "split_combo": pair["split_combo"],
                    "component": int(pair["component"]),
                    "dy": dy,
                    "df": df,
                    "delta_error": df - dy,
                    "abs_delta_error": abs(df - dy),
                    "abs_dy": abs(dy),
                    "sign_ok": sign_ok,
                }
            )
        out["pairs"][name] = rows
        strata: dict[str, Any] = {}
        for combo in sorted({r["split_combo"] for r in rows}):
            strata[combo] = _pair_metrics([r for r in rows if r["split_combo"] == combo])
        strata["ALL"] = _pair_metrics(rows)
        strata["mmp_main_H"] = _pair_metrics([r for r in rows if r["split_combo"] in ("valid-valid", "valid-test", "test-test")])
        out["per_model"][name] = strata
    # zero baseline is model independent
    all_rows = next(iter(out["pairs"].values()), [])
    if all_rows:
        out["zero_baseline"] = _pair_metrics(
            [{**r, "abs_delta_error": r["abs_dy"], "delta_error": r["dy"], "sign_ok": None} for r in all_rows]
        )
    return out


def _model_specs() -> list[tuple[str, str, str, str]]:
    specs = []
    for model in ("L", "G0", "Gx"):
        try:
            specs.append((model, C.STAGE1, model, S.selected_recipe("ogbg-mollipo", model)))
        except FileNotFoundError:
            pass
    try:
        specs.append(("A", C.STAGE2, "A", S.selected_recipe("ogbg-mollipo", "A", stage=C.STAGE2)))
    except FileNotFoundError:
        pass
    try:
        kind = S.selected_trunk_model()
        specs.append(("T", C.STAGE1, kind, S.selected_recipe("ogbg-mollipo", kind)))
        specs.append(("D", C.STAGE3, "D", "D600"))
    except FileNotFoundError:
        pass
    return specs


# ---------------------------------------------------------------------------
# stage 2/3: additive and frozen-head diagnostics
# ---------------------------------------------------------------------------


def stage23_table() -> dict[str, Any]:
    out: dict[str, Any] = {"A": {}, "D": {}}
    for dataset in C.DATASETS:
        try:
            recipe = S.selected_recipe(dataset, "A", stage=C.STAGE2)
        except FileNotFoundError:
            continue
        traj = load_trajectory(C.STAGE2, dataset, "A", recipe)
        pred = load_predictions(C.STAGE2, dataset, "A", recipe)
        kind = S.selected_trunk_model()
        t_recipe = S.selected_recipe(dataset, kind)
        t_traj = load_trajectory(C.STAGE1, dataset, kind, t_recipe)
        t_pred = load_predictions(C.STAGE1, dataset, kind, t_recipe)
        entry = {
            "recipe": recipe,
            "selected": traj["selected"],
            "parameters": traj.get("parameters"),
            "budget": traj.get("budget"),
            "head_budget": traj.get("init", {}).get("budget"),
            "trunk": kind,
            "T_recipe": t_recipe,
            "T_selected": t_traj["selected"],
            "trunk_n_parameters": t_traj["n_parameters"],
            "comparisons": {},
        }
        for split in ("valid", "test"):
            _, blocks = _align([pred[split], t_pred[split]], split)
            stats = bootstrap_rmse_diff(blocks[0]["pred_raw"], blocks[1]["pred_raw"], blocks[0]["y_raw"])
            entry["comparisons"][split] = {"a": "A", "b": "T", **stats}
        out["A"][dataset] = entry
        try:
            d_traj = load_trajectory(C.STAGE3, dataset, "D", "D600")
            d_pred = load_predictions(C.STAGE3, dataset, "D", "D600")
        except FileNotFoundError:
            out["D"][dataset] = {"status": "MISSING"}
            continue
        d_entry = {
            "selected": d_traj["selected"],
            "T": d_traj.get("T"),
            "T_recipe": d_traj.get("T_recipe"),
            "n_parameters": d_traj.get("n_parameters"),
            "comparisons": {},
        }
        for split in ("valid", "test"):
            _, blocks = _align([d_pred[split], t_pred[split]], split)
            stats = bootstrap_rmse_diff(blocks[0]["pred_raw"], blocks[1]["pred_raw"], blocks[0]["y_raw"])
            fid = rmse(blocks[0]["pred_std"], blocks[1]["pred_std"])
            d_entry["comparisons"][split] = {"a": "D", "b": "T", "teacher_fidelity_rmse_std": fid, **stats}
        out["D"][dataset] = d_entry
    return out


# ---------------------------------------------------------------------------
# oracle-calibrated explanation error
# ---------------------------------------------------------------------------


def load_oracle_truth() -> dict[str, Any]:
    path = C.ANALYSIS_DIR / "oracle_contrib" / "ogbg-mollipo.npz"
    z = np.load(path)
    return {key: z[key] for key in z.files}


def _truth_groups(truth: Mapping[str, np.ndarray], prop: str, pair: Mapping[str, Any]) -> dict[str, float]:
    c = truth[f"{prop}_atom"]
    ptr = truth["atom_ptr"]
    a, bb = int(pair["row_a"]), int(pair["row_b"])
    ca = c[ptr[a] : ptr[a + 1]]
    cb = c[ptr[bb] : ptr[bb + 1]]
    var_a = [int(x) for x in pair["var_atoms_a"]]
    var_b = [int(x) for x in pair["var_atoms_b"]]
    core_a = [int(x) for x in pair["core_atoms_a"]]
    cmap = {int(k): int(v) for k, v in pair["core_atom_map"].items()}
    return {
        "variable": float(cb[var_b].sum() - ca[var_a].sum()),
        "core": float(sum(cb[cmap[u]] - ca[u] for u in core_a)),
        "total": float((cb[var_b].sum() - ca[var_a].sum()) + sum(cb[cmap[u]] - ca[u] for u in core_a)),
    }


def _oracle_sigma(prop: str) -> float:
    b = D.oracle_bundle("ogbg-mollipo", prop)
    return float(b.scaler["std"])


def _ledger_path(tag: str):
    return C.LEDGER_DIR / f"explain_ogbg-mollipo_{tag}.json"


def _oracle_pair_predictions(prop: str, truth: Mapping[str, np.ndarray], sigma: float) -> dict[str, Any]:
    catalog = load_catalog()
    b = D.oracle_bundle("ogbg-mollipo", prop)
    out: dict[str, Any] = {}
    for model in ("L", "T", "A"):
        try:
            pred = load_predictions(C.STAGE_ORACLE, "ogbg-mollipo", model, "selected")
        except FileNotFoundError:
            out[model] = {"status": "MISSING"}
            continue
        row_map = {}
        for split in C.SPLITS:
            for row, value in zip(pred[split]["row_id"].tolist(), pred[split]["pred_raw"].tolist()):
                row_map[int(row)] = float(value)
        rows = []
        for pair in catalog:
            a, bb = int(pair["row_a"]), int(pair["row_b"])
            t = _truth_groups(truth, prop, pair)
            df = row_map[bb] - row_map[a]
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "component": int(pair["component"]),
                    "dtruth": t["total"],
                    "df": df,
                    "abs_error": abs(df - t["total"]),
                    "sign_ok": None if abs(t["total"]) < 0.1 else bool(np.sign(df) == np.sign(t["total"])),
                }
            )
        comp: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            comp[r["component"]].append(r["abs_error"])
        signs = [r for r in rows if r["sign_ok"] is not None]
        out[model] = {
            "n_pairs": len(rows),
            "delta_mae_raw": float(np.mean([r["abs_error"] for r in rows])),
            "delta_mae_sigma": float(np.mean([r["abs_error"] for r in rows]) / sigma),
            "delta_component_equal_sigma": float(np.mean([np.mean(v) for v in comp.values()]) / sigma),
            "n_sign": len(signs),
            "delta_sign_accuracy": float(np.mean([r["sign_ok"] for r in signs])) if signs else None,
            **{
                key: value
                for key, value in _oracle_traj_metrics(model).items()
                if key in ("valid_rmse_raw", "test_rmse_raw")
            },
            "valid_rmse_sigma": _oracle_traj_metrics(model).get("valid_rmse_raw", float("nan")) / sigma,
            "test_rmse_sigma": _oracle_traj_metrics(model).get("test_rmse_raw", float("nan")) / sigma,
        }
    return out


def _oracle_traj_metrics(model: str) -> dict[str, float]:
    path = C.task_dir(C.STAGE_ORACLE, "ogbg-mollipo", model, "selected") / "trajectory.json"
    if not path.is_file():
        return {}
    return C.read_json(path)["selected"]


def oracle_explanation_table() -> dict[str, Any]:
    truth = load_oracle_truth()
    catalog = {p["pair_id"]: p for p in load_catalog()}
    sample = C.read_json(C.ANALYSIS_DIR / "explain_sample.json")["pairs"]
    pair_ids = [s["pair_id"] for s in sample]
    out: dict[str, Any] = {"properties": {}, "sigma": {}, "pairs": {}}
    for prop in C.ORACLE_PROPERTIES:
        out["sigma"][prop] = _oracle_sigma(prop)
        entry = {"models": {}, "controls": {}}
        truth_rows = []
        for pid in pair_ids:
            pair = catalog[pid]
            t = _truth_groups(truth, prop, pair)
            truth_rows.append({"pair_id": pid, "component": int(pair["component"]), **t})
        entry["truth"] = truth_rows
        for model, block in _oracle_pair_predictions(prop, truth, out["sigma"][prop]).items():
            entry.setdefault("prediction", {})[model] = block
        # zero predictor and oracle-delta predictor controls
        sigma = out["sigma"][prop]
        entry["controls"]["zero"] = {
            "E": float(np.mean([(abs(r["variable"]) + abs(r["core"])) / sigma for r in truth_rows])),
            "note": "predicts zero contribution for every pair",
        }
        entry["controls"]["generative_rule"] = {
            "E": 0.0,
            "note": "pipeline positive control: scores are the true c_v, so E is zero by construction",
        }
        entry["controls"]["all_on_variable"] = {
            "E": float(np.mean([(abs(r["total"] - r["variable"]) + abs(r["core"])) / sigma for r in truth_rows])),
            "note": "uses the correct total Δy but assigns it all to the variable channel (no Δy-accuracy credit for attribution)",
        }
        entry["controls"]["conservation_check"] = {
            "max_abs_var_plus_core_minus_total": float(max(abs(r["variable"] + r["core"] - r["total"]) for r in truth_rows)),
        }
        for model in ("L", "T", "A"):
            path = _ledger_path(f"{model}_{prop}")
            if not path.is_file():
                entry["models"][model] = {"status": "MISSING"}
                continue
            ledger = C.read_json(path)
            sources = ["native"] if model == "L" else (["ig"] if model == "T" else ["native", "ig"])
            block = {}
            for source in sources:
                rows = _oracle_rows(ledger, truth_rows, sigma, source)
                if rows:
                    block[source] = _oracle_aggregate(rows)
            block["ledger"] = str(path.relative_to(C.ROUND_DIR))
            block["quadrature"] = ledger.get("quadrature")
            block["scope"] = ledger.get("scope")
            entry["models"][model] = block
        ratio = _oracle_ratio(entry, prop)
        if ratio:
            entry["A_vs_T"] = ratio
        out["properties"][prop] = entry
    return out


def _oracle_ratio(entry: Mapping[str, Any], prop: str) -> dict[str, Any] | None:
    """Component bootstrap of E_A(native) / E_T(ig) on the shared sample."""
    a_block = (entry.get("models", {}).get("A") or {}).get("native")
    t_block = (entry.get("models", {}).get("T") or {}).get("ig")
    if not a_block or not t_block:
        return None
    a_rows = {r["pair_id"]: r for r in a_block.get("rows", [])}
    t_rows = {r["pair_id"]: r for r in t_block.get("rows", [])}
    ids = sorted(set(a_rows) & set(t_rows))
    if len(ids) < 4:
        return None
    comps = sorted({int(a_rows[i]["component"]) for i in ids})
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    ratios = np.empty(BOOT_N)
    for k in range(BOOT_N):
        pick = rng.choice(comps, size=len(comps), replace=True)
        ea, et = [], []
        for c in pick:
            members = [i for i in ids if int(a_rows[i]["component"]) == c]
            ea.extend(a_rows[i]["E"] for i in members)
            et.extend(t_rows[i]["E"] for i in members)
        ratios[k] = float(np.mean(ea) / np.mean(et)) if et else float("nan")
    return {
        "property": prop,
        "n_pairs": len(ids),
        "n_components": len(comps),
        "E_A_native": float(np.mean([a_rows[i]["E"] for i in ids])),
        "E_T_ig": float(np.mean([t_rows[i]["E"] for i in ids])),
        "ratio": float(np.mean([a_rows[i]["E"] for i in ids]) / np.mean([t_rows[i]["E"] for i in ids])),
        "ratio_ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
        "prob_A_better": float(np.mean(ratios < 1.0)),
        "bootstrap": BOOT_N,
        "seed": BOOT_SEED,
        "support": "LOW_SUPPORT" if len(comps) < 30 else "OK",
    }


def _scores_from_ledger(entry: Mapping[str, Any], source: str) -> dict[str, Any] | None:
    """Return variable/core/size/global scores (standardised units)."""
    if source == "native":
        nat = entry.get("native")
        if not nat:
            return None
        return {
            "variable": float(nat["variable"]),
            "core": float(nat["core"]),
            "size": float(nat["size"]),
            "global": {k: float(v) for k, v in nat["global"].items()},
            "source": "native",
        }
    ig = entry.get("ig")
    if not ig:
        return None
    return {
        "variable": float(ig["variable"]),
        "core": float(ig["core"]),
        "size": float(ig["size"]),
        "global": {k: float(v) for k, v in ig["global"].items()},
        "source": "ig",
    }


def _oracle_rows(
    ledger: Mapping[str, Any], truth_rows: Sequence[Mapping[str, Any]], sigma: float, source: str
) -> list[dict[str, Any]]:
    by_id = {r["pair_id"]: r for r in ledger["pairs"]}
    rows = []
    for t in truth_rows:
        entry = by_id.get(t["pair_id"])
        if entry is None:
            continue
        scores = _scores_from_ledger(entry, source)
        if scores is None:
            continue
        var_s = scores["variable"] * sigma
        core_s = scores["core"] * sigma
        size_s = scores["size"] * sigma
        glob = {k: v * sigma for k, v in scores["global"].items()}
        E = (abs(var_s - t["variable"]) + abs(core_s - t["core"]) + abs(size_s) + sum(abs(v) for v in glob.values())) / sigma
        rows.append(
            {
                "pair_id": t["pair_id"],
                "component": int(t["component"]),
                "E": float(E),
                "variable": var_s,
                "core": core_s,
                "size": size_s,
                "global_abs_sum": sum(abs(v) for v in glob.values()),
                "truth_variable": t["variable"],
                "truth_core": t["core"],
                "source": scores["source"],
            }
        )
    return rows


def _oracle_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "EMPTY"}
    comp: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        comp[int(r["component"])].append(r["E"])
    return {
        "n_pairs": len(rows),
        "n_components": len(comp),
        "E_pair_mean": float(np.mean([r["E"] for r in rows])),
        "E_component_equal": float(np.mean([np.mean(v) for v in comp.values()])),
        "var_mae": float(np.mean([abs(r["variable"] - r["truth_variable"]) for r in rows])),
        "core_mae": float(np.mean([abs(r["core"] - r["truth_core"]) for r in rows])),
        "size_abs_mean": float(np.mean([abs(r["size"]) for r in rows])),
        "global_abs_mean": float(np.mean([r["global_abs_sum"] for r in rows])),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# local-effect consistency of the explained variable channel (task brief 7.9)
# ---------------------------------------------------------------------------


def mmp_explanation_table() -> dict[str, Any]:
    catalog = {p["pair_id"]: p for p in load_catalog()}
    sample = C.read_json(C.ANALYSIS_DIR / "explain_sample.json")["pairs"]
    b = D.bundle("ogbg-mollipo")
    y = row_targets(b)
    out: dict[str, Any] = {"models": {}, "n_pairs": len(sample)}
    for path in sorted(C.LEDGER_DIR.glob("explain_ogbg-mollipo_*.json")):
        tag = path.stem.replace("explain_ogbg-mollipo_", "")
        if "_" in tag:
            continue  # oracle ledgers are handled by the oracle table
        ledger = C.read_json(path)
        sigma = float(ledger.get("sigma") or b.scaler["std"])
        for source in ("native", "ig"):
            rows = []
            for entry in ledger["pairs"]:
                scores = _scores_from_ledger(entry, source)
                if scores is None:
                    continue
                pair = catalog[entry["pair_id"]]
                dy = y[int(pair["row_b"])] - y[int(pair["row_a"])]
                var_raw = scores["variable"] * sigma
                rows.append(
                    {
                        "pair_id": entry["pair_id"],
                        "component": int(pair["component"]),
                        "dy": dy,
                        "variable": var_raw,
                        "abs_error": abs(var_raw - dy),
                        "sign_ok": None if abs(dy) < 0.1 else bool(np.sign(var_raw) == np.sign(dy)),
                    }
                )
            if not rows:
                continue
            signs = [r for r in rows if r["sign_ok"] is not None]
            comp: dict[int, list[float]] = defaultdict(list)
            for r in rows:
                comp[r["component"]].append(r["abs_error"])
            x = np.asarray([r["variable"] for r in rows])
            z = np.asarray([r["dy"] for r in rows])
            out["models"][f"{tag}/{source}"] = {
                "n": len(rows),
                "variable_vs_dy_mae": float(np.mean([r["abs_error"] for r in rows])),
                "component_equal_mae": float(np.mean([np.mean(v) for v in comp.values()])),
                "sign_accuracy": float(np.mean([r["sign_ok"] for r in signs])) if signs else None,
                "n_sign": len(signs),
                "pearson_r": float(np.corrcoef(x, z)[0, 1]) if len(rows) > 2 else None,
                "dy_mean_abs": float(np.mean([abs(r["dy"]) for r in rows])),
                "support": "LOW_SUPPORT" if len(comp) < 30 else "OK",
            }
    return out


# ---------------------------------------------------------------------------
# explanation checks + algebraic reallocation
# ---------------------------------------------------------------------------


def reallocation_stress() -> dict[str, Any]:
    """7.10: equivalent native ledger with a zero-sum node reallocation."""
    out: dict[str, Any] = {}
    for model in ("A", "D"):
        path = _ledger_path(model)
        if not path.is_file():
            continue
        ledger = C.read_json(path)
        rows = []
        for entry in ledger["pairs"]:
            nat = entry.get("native")
            if not nat:
                continue
            rows.append(entry)
        out[model] = {"n_pairs": len(rows), "status": "NEEDS_NODE_ATOMS"}
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(C.REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run_manifest() -> dict[str, Any]:
    costs = C.read_json(C.ANALYSIS_DIR / "task_costs.json") if (C.ANALYSIS_DIR / "task_costs.json").is_file() else {"units": {}}
    units = []
    for task in C.all_tasks():
        path = C.task_status_path(task)
        status = C.read_json(path) if path.is_file() else {"state": "MISSING"}
        units.append(
            {
                "key": task["key"],
                "stage": task["stage"],
                "dataset": task["dataset"],
                "model": task["model"],
                "recipe": task["recipe"],
                "epochs": task["epochs"],
                "status": status.get("state", "UNKNOWN"),
                "selected_epoch": status.get("selected_epoch"),
                "signature": status.get("signature"),
                "seconds": (costs.get("units", {}).get(task["key"], {}) or {}).get("seconds"),
                "n_parameters": (costs.get("units", {}).get(task["key"], {}) or {}).get("n_parameters"),
            }
        )
    payload = {
        "round": C.ROUND_ID,
        "design_basis_commit": C.DESIGN_BASIS_COMMIT,
        "commit": _git_commit(),
        "environment": C.environment(),
        "units": units,
        "total_seconds": float(sum((u["seconds"] or 0.0) for u in units)),
        "gpu_hours_two_cards": float(sum((u["seconds"] or 0.0) for u in units)) / 3600.0,
    }
    C.write_json(C.ANALYSIS_DIR / "run_manifest.json", payload)
    return payload


def build_all() -> dict[str, Any]:
    summary: dict[str, Any] = {"started_at": time_stamp()}
    summary["manifest"] = {k: v for k, v in run_manifest().items() if k != "units"}
    stage1 = stage1_table()
    write_table("stage1", stage1, stage1["rows"])
    summary["stage1"] = {k: v for k, v in stage1.items() if k != "rows"}
    stage23 = stage23_table()
    write_table("stage23", stage23)
    summary["stage23"] = stage23
    mmp = mmp_predictions()
    write_table("mmp_predictions", {k: v for k, v in mmp.items() if k != "pairs"})
    expl = mmp_explanation_table()
    write_table("mmp_explanation", expl)
    summary["mmp"] = {k: v for k, v in mmp.items() if k != "pairs"}
    summary["finished_at"] = time_stamp()
    C.write_json(C.ANALYSIS_DIR / "analysis_summary.json", summary)
    return summary


def time_stamp() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


if __name__ == "__main__":
    job = sys.argv[1] if len(sys.argv) > 1 else "all"
    if job == "all":
        print(json.dumps(build_all(), indent=1, default=str)[:4000])
    elif job == "oracle":
        out = oracle_explanation_table()
        write_table("oracle", out)
        print(json.dumps({k: v for k, v in out.items() if k != "pairs"}, indent=1, default=str)[:4000])
    else:
        raise SystemExit(f"unknown job {job}")
