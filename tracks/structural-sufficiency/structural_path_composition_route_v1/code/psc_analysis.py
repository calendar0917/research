"""Analysis for the PSC route: tables, selection, bootstrap, pairs, ledgers.

Usage (stages are independent, ``all`` runs what is available):

    python psc_analysis.py --stage table
    python psc_analysis.py --stage select
    python psc_analysis.py --stage bootstrap
    python psc_analysis.py --stage pairs
    ...

Every number is recomputed from frozen per-run predictions; manifests are used
only for provenance and epochs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

import psc_common as C
import psc_data as D
import psc_model as M
import psc_paths as PP

RESULTS = C.ROUND_DIR / "results"
ANALYSIS = C.ROUND_DIR / "analysis"
CPU_OUT = RESULTS / "cpu"

PSC_ARMS = ("PSC_SUM", "PSC_SUM_PAIR", "PSC_SUM_BAG", "PSC_SUM_BAG_PAIR",
            "PSC_POOL", "PSC_POOL_PAIR", "PSC_POOL_BAG", "PSC_POOL_BAG_PAIR")
ARM_SPEC = {
    "PSC_SUM": dict(family="PSC", input="ORDER", readout="SUM", target="MOLECULE"),
    "PSC_SUM_PAIR": dict(family="PSC", input="ORDER", readout="SUM", target="PAIR"),
    "PSC_SUM_BAG": dict(family="PSC", input="BAG", readout="SUM", target="MOLECULE"),
    "PSC_SUM_BAG_PAIR": dict(family="PSC", input="BAG", readout="SUM", target="PAIR"),
    "PSC_POOL": dict(family="PSC", input="ORDER", readout="POOL", target="MOLECULE"),
    "PSC_POOL_PAIR": dict(family="PSC", input="ORDER", readout="POOL", target="PAIR"),
    "PSC_POOL_BAG": dict(family="PSC", input="BAG", readout="POOL", target="MOLECULE"),
    "PSC_POOL_BAG_PAIR": dict(family="PSC", input="BAG", readout="POOL", target="PAIR"),
    "PSC_SUM_DIAG": dict(family="PSC", input="ORDER", readout="SUM", target="DIAG"),
    "PSC_SUM_BAG_DIAG": dict(family="PSC", input="BAG", readout="SUM", target="DIAG"),
    "PSC_POOL_DIAG": dict(family="PSC", input="ORDER", readout="POOL", target="DIAG"),
    "PSC_POOL_BAG_DIAG": dict(family="PSC", input="BAG", readout="POOL", target="DIAG"),
    "ACRA_REF": dict(family="ACRA", input="site/context events", readout="ACRA",
                     target="parent"),
    "ACRA_PAIR": dict(family="ACRA", input="site/context events", readout="ACRA",
                      target="parent+PAIR"),
    "GINE_REF": dict(family="GINE", input="raw graph", readout="GINE", target="MOLECULE"),
}
CPU_MODELS = ("PATH_COUNT_RIDGE", "PATH_COUNT_KRR", "MORGAN_COUNT_KRR", "COUNT_ONLY_RIDGE")


def run_id(arm: str, split: str, seed: int) -> str:
    return {"D0": f"{arm}_S{seed}", "D1": f"{arm}_D1_S{seed}",
            "FULL": f"{arm}_FULL_S{seed}"}[split]


def load_manifest(rid: str) -> dict[str, Any] | None:
    path = RESULTS / rid / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def eval_prediction(rid: str, split: str) -> dict[str, np.ndarray] | None:
    """Primary evaluation predictions (best checkpoint on D0, final on D1/FULL)."""
    tag = "best" if split == "D0" else "final"
    path = RESULTS / rid / f"predictions_{tag}_eval.npz"
    if not path.is_file():
        path = RESULTS / rid / "predictions_final_eval.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def fit_prediction(rid: str, split: str) -> dict[str, np.ndarray] | None:
    tag = "best" if split == "D0" else "final"
    path = RESULTS / rid / f"predictions_{tag}_fit.npz"
    if not path.is_file():
        path = RESULTS / rid / "predictions_final_fit.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------


def collect_records(bundle: D.DataBundle) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = bundle.labels
    records: list[dict[str, Any]] = []
    preds: dict[str, dict[str, Any]] = {}
    for arm, spec in ARM_SPEC.items():
        for split, seeds in (("D0", (0, 1, 2)), ("D1", (0, 1, 2)), ("FULL", (0, 1, 2))):
            for seed in seeds:
                rid = run_id(arm, split, seed)
                man = load_manifest(rid)
                base = {"arm": arm, "run_id": rid, "split": split, "seed": seed,
                        **{k: spec[k] for k in ("family", "input", "readout", "target")}}
                if man is None:
                    records.append({**base, "status": "MISSING"})
                    continue
                pred = eval_prediction(rid, split)
                if pred is None:
                    records.append({**base, "status": "NO_PREDICTIONS"})
                    continue
                rows = pred["row_id"]
                if split == "FULL":
                    records.append({**base, "status": "PENDING_FINAL_EVAL",
                                    "n_eval": int(rows.size),
                                    "wall_seconds": man.get("wall_seconds"),
                                    "gpu_hours": man.get("gpu_hours"),
                                    "commit": man.get("provenance", {}).get("git_commit"),
                                    "params_total": man.get("params", {}).get("total"),
                                    "epochs": int(man.get("final", {}).get("epoch", 0)),
                                    "best_epoch": (man.get("best") or {}).get("epoch")})
                    preds[rid] = {"row_id": rows, "pred": pred["pred"], "split": split}
                    continue
                y = labels[rows]
                p = np.asarray(pred["pred"], np.float64)
                best = man.get("best") or {}
                fin = man.get("final") or {}
                records.append({**base, "status": "OK",
                                "n_eval": int(rows.size),
                                "rmse": C.rmse(p, y), "mae": C.mae(p, y),
                                "bias": float(np.mean(p - y)),
                                "checkpoint": "best(D0 dev RMSE)" if split == "D0" else "final epoch",
                                "best_epoch": best.get("epoch"),
                                "last_epoch": fin.get("epoch"),
                                "last_eval_rmse": fin.get("eval_rmse"),
                                "wall_seconds": man.get("wall_seconds"),
                                "gpu_hours": man.get("gpu_hours"),
                                "params_total": man.get("params", {}).get("total"),
                                "max_vram_bytes": man.get("max_vram_bytes"),
                                "commit": man.get("provenance", {}).get("git_commit"),
                                "split_sha256": man.get("split", {}).get("sha256"),
                                "fit_rows_sha256": man.get("split", {}).get("fit_rows_sha256")})
                preds[rid] = {"row_id": rows, "pred": p, "split": split}
    # CPU baselines
    for model in CPU_MODELS:
        for split in ("D0", "D1", "FULL"):
            rid = f"CPU_{model}_{split}"
            man = load_manifest(f"cpu/{rid}") if False else None
            path = CPU_OUT / rid / "manifest.json"
            if not path.is_file():
                records.append({"arm": model, "run_id": rid, "split": split, "seed": 0,
                                "family": "CPU", "input": "path counts" if "PATH" in model
                                else ("morgan counts" if "MORGAN" in model else "counts"),
                                "readout": "closed form", "target": "MOLECULE",
                                "status": "MISSING"})
                continue
            man = json.loads(path.read_text(encoding="utf-8"))
            pred = np.load(CPU_OUT / rid / "predictions.npz", allow_pickle=False)
            rows = np.asarray(pred["row_id"])
            p = np.asarray(pred["pred"])
            base = {"arm": model, "run_id": rid, "split": split, "seed": 0,
                    "family": "CPU", "input": man.get("extra", {}).get("features", "counts")
                    if isinstance(man.get("extra"), dict) else "counts",
                    "readout": "closed form", "target": "MOLECULE", "status": "OK",
                    "n_eval": int(rows.size), "wall_seconds": man.get("wall_seconds"),
                    "gpu_hours": 0.0, "params_total": None,
                    "selected": man.get("selected"), "n_configs": man.get("n_configs"),
                    "commit": man.get("provenance", {}).get("git_commit")}
            if split == "FULL":
                records.append({**base, "status": "PENDING_FINAL_EVAL"})
            else:
                y = labels[rows]
                records.append({**base, "rmse": C.rmse(p, y), "mae": C.mae(p, y),
                                "bias": float(np.mean(p - y)),
                                "checkpoint": "closed form (alpha selected on D0)"
                                if split == "D0" else "locked hyper-parameters"})
            preds[rid] = {"row_id": rows, "pred": p, "split": split}
    return records, preds


def aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("D0", "D1"):
        agg: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for r in records:
            if r["split"] == split and r["status"] == "OK":
                agg[r["arm"]].append(r)
        out[split] = {}
        for arm, rs in agg.items():
            rms = np.asarray([r["rmse"] for r in sorted(rs, key=lambda x: x["seed"])])
            maes = np.asarray([r["mae"] for r in sorted(rs, key=lambda x: x["seed"])])
            out[split][arm] = {
                "n_seeds": int(rms.size), "rmse_mean": float(rms.mean()),
                "rmse_sd": float(rms.std(ddof=1)) if rms.size > 1 else 0.0,
                "rmse_seeds": rms.tolist(), "mae_mean": float(maes.mean()),
                "mae_sd": float(maes.std(ddof=1)) if maes.size > 1 else 0.0,
                "mae_seeds": maes.tolist(),
                "best_epochs": [r.get("best_epoch") for r in sorted(rs, key=lambda x: x["seed"])],
                "wall_seconds": float(sum(r.get("wall_seconds") or 0 for r in rs)),
                "gpu_hours": float(sum(r.get("gpu_hours") or 0 for r in rs)),
            }
    return out


def cmd_table(bundle: D.DataBundle) -> dict[str, Any]:
    records, preds = collect_records(bundle)
    agg = aggregate(records)
    base = agg["D0"].get("ACRA_REF", {}).get("rmse_mean")
    if base is None:
        base = agg["D0"].get("ACRA_PAIR", {}).get("rmse_mean")
    rows = []
    for r in records:
        row = dict(r)
        if r["status"] == "OK" and base is not None:
            row["delta_vs_ACRA_REF_D0_mean"] = None
            agg_arm = agg["D0"].get(r["arm"])
            if r["split"] == "D0" and agg_arm:
                row["delta_vs_ACRA_REF_D0_mean"] = base - agg_arm["rmse_mean"]
        rows.append(row)
    C.write_csv(RESULTS / "main_table.csv", rows)
    summary = {"generated_utc": C.now_utc(), "git_commit": C.git_commit(),
               "aggregate": agg, "n_records": len(rows),
               "gpu_hours_total": float(sum(r.get("gpu_hours") or 0 for r in records)),
               "notes": {"D0": "best checkpoint by D0 dev whole-molecule RMSE",
                         "D1": "final epoch, fixed E_m, no selection",
                         "FULL": "final epoch; official Valid scored once after freeze",
                         "cpu": "closed form; alpha/radius locked from D0"}}
    C.write_json(RESULTS / "summary.json", summary)
    print(json.dumps({s: {a: round(v["rmse_mean"], 4) for a, v in agg[s].items()}
                      for s in agg}, indent=1))
    return {"records": records, "aggregate": agg, "preds": preds}


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def _mean(agg: Mapping[str, Any], arm: str) -> float | None:
    v = agg.get(arm)
    return None if v is None else float(v["rmse_mean"])


def choose_psc(agg_d0: Mapping[str, Any], tolerance: float) -> dict[str, Any] | None:
    vals = {a: _mean(agg_d0, a) for a in PSC_ARMS}
    vals = {a: v for a, v in vals.items() if v is not None}
    if len(vals) < len(PSC_ARMS):
        return None
    r_min = min(vals.values())
    cand = [a for a, v in vals.items() if v <= r_min + tolerance]
    readout_rank = {"SUM": 0, "POOL": 1}
    target_rank = {"MOLECULE": 0, "PAIR": 1}
    input_rank = {"ORDER": 0, "BAG": 1}
    cand.sort(key=lambda a: (readout_rank[ARM_SPEC[a]["readout"]],
                             target_rank[ARM_SPEC[a]["target"]],
                             input_rank[ARM_SPEC[a]["input"]], a))
    return {"champion": cand[0], "r_min": r_min, "tolerance": tolerance,
            "tolerance_set": sorted(cand), "all_means": vals,
            "rule": "min of 3-seed mean D0 RMSE; S = {mean <= r_min+0.005}; "
                    "priority SUM>POOL, MOLECULE>PAIR, ORDER>BAG, then id"}


def twin(arm: str, field: str) -> str | None:
    """Arm with one factor flipped (input or readout)."""
    spec = ARM_SPEC[arm]
    for cand, cspec in ARM_SPEC.items():
        if cspec["target"] != spec["target"] or cspec["family"] != "PSC":
            continue
        if field == "input" and cspec["readout"] == spec["readout"] \
                and cspec["input"] != spec["input"]:
            return cand
        if field == "readout" and cspec["input"] == spec["input"] \
                and cspec["readout"] != spec["readout"]:
            return cand
    return None


def diag_twin(arm: str) -> str | None:
    spec = ARM_SPEC[arm]
    if spec["family"] != "PSC" or spec["target"] != "PAIR":
        return None
    for cand, cspec in ARM_SPEC.items():
        if (cspec["family"] == "PSC" and cspec["target"] == "DIAG"
                and cspec["readout"] == spec["readout"]
                and cspec["input"] == spec["input"]):
            return cand
    return None


def cmd_select(bundle: D.DataBundle) -> dict[str, Any]:
    data = cmd_table(bundle)
    agg0 = data["aggregate"]["D0"]
    cfg = C.load_config()
    tol = float(cfg["selection"]["tolerance_rmse"])
    pick = choose_psc(agg0, tol)
    if pick is None:
        raise SystemExit("selection needs all eight PSC arms with 3 seeds on D0")
    champ = pick["champion"]
    acra_vals = {a: _mean(agg0, a) for a in ("ACRA_REF", "ACRA_PAIR")}
    acra_vals = {a: v for a, v in acra_vals.items() if v is not None}
    if len(acra_vals) < 2:
        raise SystemExit("ACRA_B* needs ACRA_REF and ACRA_PAIR on D0")
    a_min = min(acra_vals.values())
    acra_cand = [a for a, v in acra_vals.items() if v <= a_min + tol]
    acra_b = "ACRA_REF" if "ACRA_REF" in acra_cand else "ACRA_PAIR"
    o_star = twin(champ, "input")
    r_star = twin(champ, "readout")
    m_star = o_star if ARM_SPEC[champ]["readout"] == "SUM" else r_star
    d_twin = diag_twin(champ) if ARM_SPEC[champ]["target"] == "PAIR" else None
    e_m = {}
    for arm in {champ, o_star, r_star, acra_b, "GINE_REF"} | ({d_twin} if d_twin else set()):
        if arm is None:
            continue
        v = agg0.get(arm)
        epochs = [e for e in (v["best_epochs"] if v else []) if e]
        e_m[arm] = int(round(float(np.median(epochs)))) if epochs else None
    cpu_lock = {}
    for model in CPU_MODELS:
        path = CPU_OUT / f"CPU_{model}_D0" / "manifest.json"
        if path.is_file():
            man = json.loads(path.read_text(encoding="utf-8"))
            sel = man.get("selected", {})
            cpu_lock[model] = {"alpha": sel.get("alpha"), "radius": sel.get("radius", 0),
                               "config_key": sel.get("config_key"),
                               "rmse_D0": sel.get("rmse"), "mae_D0": sel.get("mae")}
    payload = {
        "generated_utc": C.now_utc(), "git_commit": C.git_commit(),
        "rule": pick["rule"],
        "psc_selection": pick,
        "champion": champ, "champion_spec": ARM_SPEC[champ],
        "O_star": o_star, "R_star": r_star, "M_star": m_star,
        "ACRA_selection": {"means": acra_vals, "ACRA_B_star": acra_b,
                           "tie_priority": ["REF", "PAIR"]},
        "DIAG_twin": d_twin,
        "requires_DIAG": bool(ARM_SPEC[champ]["target"] == "PAIR"),
        "E_m": e_m,
        "seeds_d1": [0, 1],
        "seeds_full": [0, 1, 2],
        "cpu_lock": cpu_lock,
        "d0_table": {a: agg0.get(a) for a in agg0},
        "extensions": {
            "d1_seed2": "eligible only if wall-clock and GPU budget allow before T0+6:30",
            "M_star_full": "eligible only if decided before official Valid labels are opened",
            "esol_transfer": "eligible only after the required queue, before T0+6:30",
            "size_curve": "eligible only after the required queue, before T0+6:30",
        },
        "note": "selection uses D0 only; D1 and Valid cannot change the champion",
    }
    C.write_json(ANALYSIS / "selection_lock.json", payload)
    print(json.dumps({"champion": champ, "O*": o_star, "R*": r_star, "M*": m_star,
                      "ACRA_B*": acra_b, "DIAG": d_twin, "E_m": e_m,
                      "r_min": pick["r_min"], "S": pick["tolerance_set"]}, indent=1))
    return payload


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


CONTRASTS = [
    # (candidate, reference, question)
    ("PSC_SUM_BAG", "PSC_SUM", "Q2 ORDER vs BAG (SUM, MOLECULE)"),
    ("PSC_POOL_BAG", "PSC_POOL", "Q2 ORDER vs BAG (POOL, MOLECULE)"),
    ("PSC_SUM_BAG_PAIR", "PSC_SUM_PAIR", "Q2 ORDER vs BAG (SUM, PAIR)"),
    ("PSC_POOL_BAG_PAIR", "PSC_POOL_PAIR", "Q2 ORDER vs BAG (POOL, PAIR)"),
    ("PSC_POOL", "PSC_SUM", "Q3 SUM vs POOL (ORDER, MOLECULE)"),
    ("PSC_POOL_BAG", "PSC_SUM_BAG", "Q3 SUM vs POOL (BAG, MOLECULE)"),
    ("PSC_POOL_PAIR", "PSC_SUM_PAIR", "Q3 SUM vs POOL (ORDER, PAIR)"),
    ("PSC_POOL_BAG_PAIR", "PSC_SUM_BAG_PAIR", "Q3 SUM vs POOL (BAG, PAIR)"),
    ("PSC_SUM_PAIR", "PSC_SUM", "Q4 PAIR coupling vs MOLECULE (SUM, ORDER)"),
    ("PSC_POOL_PAIR", "PSC_POOL", "Q4 PAIR coupling vs MOLECULE (POOL, ORDER)"),
    ("PSC_SUM_BAG_PAIR", "PSC_SUM_BAG", "Q4 PAIR coupling vs MOLECULE (SUM, BAG)"),
    ("PSC_POOL_BAG_PAIR", "PSC_POOL_BAG", "Q4 PAIR coupling vs MOLECULE (POOL, BAG)"),
    ("ACRA_PAIR", "ACRA_REF", "Q4 ACRA pair regulariser vs parent"),
]


def _sq_err_table(bundle: D.DataBundle, split: str, models: Sequence[str],
                  rows: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(n_models, n_seeds_max, n_rows) squared errors + availability mask."""
    sq, seeds_of, ok_models = [], [], []
    ref_rows = rows
    for m in models:
        seeds = []
        for seed in (0, 1, 2):
            if split == "D0":
                n_seeds = 3
            d = prediction_dict(m, split, seed)
            if not d:
                continue
            r = np.asarray(sorted(d.keys())) if ref_rows is None else np.asarray(ref_rows)
            if ref_rows is None:
                ref_rows = r
            if np.array_equal(np.asarray(sorted(d.keys())), np.asarray(ref_rows)) is False:
                if not set(ref_rows.tolist()) <= set(d.keys()):
                    continue
            seeds.append(np.asarray([(d[int(x)] - bundle.labels[int(x)]) ** 2 for x in ref_rows]))
        if seeds:
            sq.append(np.stack(seeds))
            seeds_of.append(m)
    if not sq:
        raise SystemExit(f"no predictions available for split {split}")
    width = max(x.shape[0] for x in sq)
    table = np.full((len(sq), width, ref_rows.size), np.nan)
    for i, x in enumerate(sq):
        table[i, :x.shape[0]] = x
    return table, np.asarray(ref_rows, np.int64), seeds_of


def cmd_bootstrap(bundle: D.DataBundle, split_name: str = "D0") -> dict[str, Any]:
    lock = C.read_json(ANALYSIS / "selection_lock.json")
    models = list(ARM_SPEC) + list(CPU_MODELS)
    table, rows, used = _sq_err_table(bundle, split_name, models)
    y = bundle.labels[rows]
    cfg = C.load_config()
    reps, seed0 = int(cfg["bootstrap"]["reps"]), int(cfg["bootstrap"]["seed"])
    n = rows.size
    n_models = table.shape[0]
    rng = np.random.default_rng(seed0)
    draw = np.empty((reps, n_models))
    for r in range(reps):
        idx = rng.integers(0, n, size=n)
        draw[r] = np.nanmean(np.sqrt(np.nanmean(table[:, :, idx], axis=2)), axis=1)
    point = np.nanmean(np.sqrt(np.nanmean(table, axis=2)), axis=1)
    index = {m: i for i, m in enumerate(used)}
    out: dict[str, Any] = {
        "generated_utc": C.now_utc(), "split": split_name, "n_rows": int(n),
        "reps": reps, "bootstrap_seed": seed0,
        "models": used,
        "point": {m: {"rmse_seed_mean": float(point[index[m]]),
                      "per_seed": [float(np.sqrt(np.mean(table[index[m], s])))
                                   for s in range(table.shape[1])
                                   if np.isfinite(table[index[m], s]).any()]}
                  for m in used},
        "definition": ("per-molecule resampling with one shared index for every model "
                       "and seed; statistic = mean over seeds of resampled RMSE"),
    }
    def add(a: str, b: str, label: str) -> dict[str, Any] | None:
        if a not in index or b not in index:
            return None
        d = draw[:, index[b]] - draw[:, index[a]]
        return {"candidate": a, "reference": b, "question": label,
                "point": float(point[index[b]] - point[index[a]]),
                "ci95": C.percentile_ci(d),
                "p_two_sided": float(2 * min(np.mean(d <= 0), np.mean(d >= 0))),
                "definition": "RMSE(reference) - RMSE(candidate); >0 candidate better"}
    contrasts = {}
    for a, b, label in CONTRASTS:
        got = add(a, b, label)
        if got:
            contrasts[f"{a}_vs_{b}"] = got
    champ = lock["champion"]
    for ref in (lock["ACRA_B_star"], "ACRA_PAIR", "GINE_REF", "PATH_COUNT_KRR",
                "MORGAN_COUNT_KRR", "COUNT_ONLY_RIDGE"):
        got = add(champ, ref, f"Q1 champion vs {ref}")
        if got:
            contrasts[f"champion_vs_{ref}"] = got
    out["contrasts"] = contrasts
    # scaffold-cluster bootstrap (D0 and D1 both have a scaffold map)
    try:
        split = C.split_spec(split_name, bundle.raw,
                             f1=(json.loads(C.SPLIT_F1.read_text(encoding="utf-8"))
                                 if split_name == "D1" else None))
        scaffold = split_rows_scaffold(bundle, split)
        groups = defaultdict(list)
        for i, sc in enumerate(scaffold):
            groups[sc].append(i)
        keys = list(groups)
        rng2 = np.random.default_rng(seed0 + 1)
        cdraw = np.empty((reps, n_models))
        for r in range(reps):
            picks = rng2.integers(0, len(keys), size=len(keys))
            idx = np.concatenate([groups[keys[k]] for k in picks])
            cdraw[r] = np.nanmean(np.sqrt(np.nanmean(table[:, :, idx], axis=2)), axis=1)
        cluster = {}
        for a, b, label in CONTRASTS + [(champ, lock["ACRA_B_star"], "Q1")]:
            if a not in index or b not in index:
                continue
            d = cdraw[:, index[b]] - cdraw[:, index[a]]
            cluster[f"{a}_vs_{b}"] = {"point": float(point[index[b]] - point[index[a]]),
                                      "ci95": C.percentile_ci(d),
                                      "n_clusters": len(keys)}
        out["scaffold_cluster_bootstrap"] = cluster
    except Exception as exc:  # pragma: no cover - scaffold map is optional
        out["scaffold_cluster_bootstrap"] = {"error": repr(exc)}
    path = ANALYSIS / ("bootstrap.json" if split_name == "D0" else f"bootstrap_{split_name}.json")
    C.write_json(path, out)
    print(json.dumps({k: {"d": round(v["point"], 4),
                          "ci": [round(x, 4) for x in v["ci95"]],
                          "p": round(v["p_two_sided"], 3)}
                      for k, v in contrasts.items()}, indent=1))
    return out


def split_rows_scaffold(bundle: D.DataBundle, split: Mapping[str, Any]) -> list[str]:
    if split["name"] == "D0":
        payload = C.load_split_f0()
        key = "scaffold_by_row"
    else:
        payload = json.loads(C.SPLIT_F1.read_text(encoding="utf-8"))
        key = "scaffold_by_row"
    scaf = payload[key]
    return [str(scaf[str(int(r))]) for r in np.asarray(split["eval_rows"], np.int64)]


# ---------------------------------------------------------------------------
# pair-level data
# ---------------------------------------------------------------------------


def prediction_dict(model: str, split: str, seed: int) -> dict[int, float]:
    """row_id -> prediction (original units) for every saved split of a run."""
    if model in ARM_SPEC:
        base = RESULTS / run_id(model, split, seed)
        tags = ("best",) if split == "D0" else ("final",)
    else:
        base = CPU_OUT / f"CPU_{model}_{split}"
        tags = ("",)
    out: dict[int, float] = {}
    for tag in tags:
        for part in ("fit", "eval"):
            name = f"predictions_{tag}_{part}.npz" if tag else "predictions.npz"
            path = base / name
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as z:
                out.update(dict(zip(np.asarray(z["row_id"]).tolist(),
                                    np.asarray(z["pred"], np.float64).tolist())))
        if out:
            break
    return out


class PairData:
    def __init__(self, bundle: D.DataBundle, split: Mapping[str, Any], log=print) -> None:
        import psc_train as T
        self.bundle = bundle
        self.split = split
        self.dir = T.ensure_pair_directory(bundle, log=log)
        self.targets = D.build_pair_targets(self.dir, split["fit_rows"], bundle.labels.size)
        fit = np.zeros(bundle.labels.size, bool)
        fit[np.asarray(split["fit_rows"], np.int64)] = True
        held = np.zeros(bundle.labels.size, bool)
        held[np.asarray(split["eval_rows"], np.int64)] = True
        a, b = self.dir.mol_a, self.dir.mol_b
        self.cut_scope = np.full(a.size, -1, np.int8)
        self.cut_scope[fit[a] & fit[b]] = 0
        self.cut_scope[(fit[a] & held[b]) | (held[a] & fit[b])] = 1
        self.cut_scope[held[a] & held[b]] = 2
        # unique unordered pairs over fit ∪ held (the pair graph used for clusters)
        keep = np.nonzero((fit[a] | held[a]) & (fit[b] | held[b]))[0]
        lo = np.minimum(a[keep], b[keep])
        hi = np.maximum(a[keep], b[keep])
        key = lo.astype(np.int64) * (bundle.labels.size + 1) + hi
        upairs, first = np.unique(key, return_index=True)
        self.cut_idx = keep[first]
        self.pa = lo[first[np.argsort(upairs)]]
        self.pb = hi[first[np.argsort(upairs)]]
        self.scope = np.where(fit[self.pa] & fit[self.pb], 0,
                              np.where(held[self.pa] & held[self.pb], 2, 1)).astype(np.int8)
        comp_nodes = _components(bundle.labels.size, self.pa, self.pb)
        self.component = comp_nodes[self.pa]
        self.fit = fit
        self.held = held

    def cut_rows_of_pair(self) -> np.ndarray:
        """Mapping unique pair index -> cut row indices (list of arrays)."""
        lo = np.minimum(self.dir.mol_a, self.dir.mol_b)
        hi = np.maximum(self.dir.mol_a, self.dir.mol_b)
        key_of_cut = lo.astype(np.int64) * (self.bundle.labels.size + 1) + hi
        pair_key = self.pa.astype(np.int64) * (self.bundle.labels.size + 1) + self.pb
        order = np.argsort(key_of_cut, kind="stable")
        keys_sorted = key_of_cut[order]
        out = []
        for k in pair_key.tolist():
            lo_i = np.searchsorted(keys_sorted, k, side="left")
            hi_i = np.searchsorted(keys_sorted, k, side="right")
            out.append(order[lo_i:hi_i])
        return out


def _components(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    parent = np.arange(int(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for x, y in zip(a.tolist(), b.tolist()):
        rx, ry = find(int(x)), find(int(y))
        if rx != ry:
            parent[rx] = ry
    return np.asarray([find(i) for i in range(int(n))], np.int64)


def delta_of(d: Mapping[int, float], pd: "PairData") -> tuple[np.ndarray, np.ndarray]:
    """Predicted pair Δ for every unique pair; NaN + mask where unavailable."""
    ok = np.asarray([int(a) in d and int(b) in d for a, b in
                     zip(pd.pa.tolist(), pd.pb.tolist())], bool)
    dp = np.full(pd.pa.size, np.nan)
    idx = np.nonzero(ok)[0]
    dp[idx] = np.asarray([d[int(pd.pb[i])] - d[int(pd.pa[i])] for i in idx.tolist()])
    return dp, ok


def pair_metrics(delta_true: np.ndarray, delta_pred: np.ndarray,
                 weights: np.ndarray | None = None) -> dict[str, Any]:
    y = np.asarray(delta_true, np.float64)
    p = np.asarray(delta_pred, np.float64)
    if y.size == 0:
        return {"n": 0}
    err = p - y
    nz = np.abs(y) > 1e-8
    w = np.ones_like(y) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    out = {
        "n": int(y.size),
        "mae": float(np.sum(w * np.abs(err))),
        "rmse": float(np.sqrt(np.sum(w * err * err))),
        "bias": float(np.sum(w * err)),
        "n_direction": int(nz.sum()),
        "direction_accuracy": (float(np.mean(np.sign(p[nz]) == np.sign(y[nz])))
                               if nz.any() else None),
        "sd_true": float(np.std(y)), "mean_abs_true": float(np.mean(np.abs(y))),
    }
    if np.std(p) > 0 and np.std(y) > 0:
        out["corr"] = float(np.corrcoef(y, p)[0, 1])
    return out


# ---------------------------------------------------------------------------
# support classes, novelty, unique pair metrics
# ---------------------------------------------------------------------------


TRC_KEYS = ("parse_mol", "parse_kind", "parse_S", "parse_C", "parse_cut_bond_row",
            "parse_S_atoms",
            "parse_S_atoms_ptr", "parse_C_atoms", "parse_C_atoms_ptr", "pair_C")


def load_trc() -> dict[str, np.ndarray]:
    with np.load(C.TRC_CACHE, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in TRC_KEYS}


def fit_fragment_set(pair_dir: Any, fit_rows: Sequence[int],
                     trc: Mapping[str, np.ndarray] | None = None) -> set[int]:
    if trc is None:
        trc = load_trc()
    parse_mol = np.asarray(trc["parse_mol"], np.int64)
    parse_kind = np.asarray(trc["parse_kind"], np.uint8)
    parse_S = np.asarray(trc["parse_S"], np.int64)
    fit = np.zeros(int(parse_mol.max()) + 1, bool)
    fit[np.asarray(fit_rows, np.int64)] = True
    use = (parse_kind == 0) & fit[parse_mol]
    return set(int(x) for x in np.unique(parse_S[use]).tolist())


def path_count_dicts(cache: PP.PathCache, rows: Sequence[int]) -> list[dict[int, int]]:
    pos = cache.positions(rows)
    out = []
    for s, e in zip(cache.mol_ptr[pos].tolist(), cache.mol_ptr[pos + 1].tolist()):
        ids = cache.ent_path[s:e].tolist()
        mult = cache.ent_mult[s:e].tolist()
        out.append(dict(zip(ids, mult)))
    return out


def bag_keys(cache: PP.PathCache) -> dict[int, bytes]:
    L = cache.L
    out = {}
    for pid in range(cache.n_paths):
        l = int(cache.path_len[pid])
        out[pid] = (cache.path_atom_bag[pid, :l + 1].tobytes()
                    + cache.path_bond_bag[pid, :l].tobytes())
    return out


def cmd_pairs(bundle: D.DataBundle) -> dict[str, Any]:
    lock = C.read_json(ANALYSIS / "selection_lock.json")
    models = [lock["champion"], lock["M_star"], lock["ACRA_B_star"], "GINE_REF"]
    if lock.get("DIAG_twin"):
        models.append(lock["DIAG_twin"])
    models += ["PATH_COUNT_KRR", "MORGAN_COUNT_KRR"]
    models = [m for m in dict.fromkeys(models) if m]
    out: dict[str, Any] = {"generated_utc": C.now_utc(), "models": models, "splits": {}}
    cfg = C.load_config()
    buckets = cfg["analysis"]["novelty_buckets"]
    for split_name in ("D0", "D1"):
        f1 = None
        if split_name == "D1":
            f1 = json.loads(C.SPLIT_F1.read_text(encoding="utf-8"))
            f1["file_sha256"] = C.sha256_file(C.SPLIT_F1)
        split = C.split_spec(split_name, bundle.raw, f1=f1)
        pd = PairData(bundle, split)
        cache_rows = np.union1d(bundle.official_train, bundle.official_valid)
        meta = PP.ensure_cache(bundle.raw, cache_rows, int(cfg["paths"]["max_len"]))
        cache = PP.PathCache(meta, device="cpu")
        # per-molecule path counts and bag counts for eta
        all_rows = np.union1d(np.asarray(split["fit_rows"]), np.asarray(split["eval_rows"]))
        counts = dict(zip(all_rows.tolist(), path_count_dicts(cache, all_rows)))
        vocab_fit = set(D.fit_path_vocab(cache, split["fit_rows"], cache.L).ids)
        bag_of = bag_keys(cache)
        bag_vocab = {bag_of[p] for p in vocab_fit}
        bag_counts = {}
        for r, dct in counts.items():
            agg: dict[bytes, int] = defaultdict(int)
            for pid, m in dct.items():
                agg[bag_of[pid]] += int(m)
            bag_counts[r] = dict(agg)
        unk = D.unseen_category_fraction(bundle.raw, all_rows,
                                         *D.category_support(
                                             bundle.raw, split["fit_rows"],
                                             *_dims()))
        trc = load_trc()
        frag_seen = fit_fragment_set(pd.dir, split["fit_rows"], trc)
        fit_T = set(np.unique(pd.dir.T_id[pd.cut_scope == 0]).tolist())
        # support classes per cut row
        cut_class = np.full(pd.dir.mol_a.size, -1, np.int8)
        seenT = np.isin(pd.dir.T_id, np.asarray(sorted(fit_T), np.int64)) if fit_T \
            else np.zeros(pd.dir.mol_a.size, bool)
        fa = np.asarray([int(w) in frag_seen for w in pd.dir.word_a.tolist()])
        fb = np.asarray([int(w) in frag_seen for w in pd.dir.word_b.tolist()])
        unseen_ends = (~fa).astype(int) + (~fb).astype(int)
        cut_class[seenT] = 0
        cut_class[~seenT & (unseen_ends == 0)] = 1
        cut_class[~seenT & (unseen_ends == 1)] = 2
        cut_class[~seenT & (unseen_ends == 2)] = 3
        # eta per unique pair
        eta = np.full(pd.pa.size, np.nan)
        eta_bag = np.full(pd.pa.size, np.nan)
        eta_len = {l: np.full(pd.pa.size, np.nan) for l in range(3, cache.L + 1)}
        for i, (a, b) in enumerate(zip(pd.pa.tolist(), pd.pb.tolist())):
            da, db = counts[a], counts[b]
            keys = set(da) | set(db)
            num = sum(abs(db.get(k, 0) - da.get(k, 0)) for k in keys if k not in vocab_fit)
            den = sum(abs(db.get(k, 0) - da.get(k, 0)) for k in keys)
            eta[i] = num / den if den > 0 else np.nan
            ba, bb = bag_counts[a], bag_counts[b]
            bkeys = set(ba) | set(bb)
            bnum = sum(abs(bb.get(k, 0) - ba.get(k, 0)) for k in bkeys if k not in bag_vocab)
            bden = sum(abs(bb.get(k, 0) - ba.get(k, 0)) for k in bkeys)
            eta_bag[i] = bnum / bden if bden > 0 else np.nan
            for l in eta_len:
                num_l = sum(abs(db.get(k, 0) - da.get(k, 0)) for k in keys
                            if k not in vocab_fit and int(cache.path_len[k]) == l)
                den_l = sum(abs(db.get(k, 0) - da.get(k, 0)) for k in keys
                            if int(cache.path_len[k]) == l)
                eta_len[l][i] = num_l / den_l if den_l > 0 else np.nan
        # molecule prediction lookups per model/seed (all rows available)
        data: dict[str, list[dict[int, float]]] = {}
        for m in models:
            lst = []
            for seed in (0, 1, 2):
                d = prediction_dict(m, split_name, seed)
                if d:
                    lst.append(d)
            if lst:
                data[m] = lst
        y = bundle.labels
        delta_true = y[pd.pb] - y[pd.pa]
        scope_names = {0: "fit_fit", 1: "fit_heldout", 2: "heldout_heldout"}
        result: dict[str, Any] = {
            "counts": {"unique_pairs": int(pd.pa.size),
                       "scope_counts": {scope_names[s]: int((pd.scope == s).sum())
                                        for s in (0, 1, 2)},
                       "components_all": int(np.unique(pd.component).size),
                       "unknown_fraction": unk},
            "scopes": {}, "support_classes": {}, "novelty": {},
            "cut_weighted": {},
        }
        cut_lists = pd.cut_rows_of_pair()
        for scope in (0, 1, 2):
            mask = pd.scope == scope
            entry: dict[str, Any] = {"n_pairs": int(mask.sum()),
                                     "n_components": int(np.unique(pd.component[mask]).size)}
            for m, seeds_pred in data.items():
                per_seed = []
                n_scored = None
                for d in seeds_pred:
                    dp, ok = delta_of(d, pd)
                    use = mask & ok
                    n_scored = int(use.sum())
                    per_seed.append(pair_metrics(delta_true[use], dp[use]))
                if not per_seed[0]["n"]:
                    entry[m] = {"n_scored": 0}
                    continue
                entry[m] = {"n_scored": n_scored, "mean_over_seeds": {
                    k: float(np.mean([ps[k] for ps in per_seed]))
                    for k in ("mae", "rmse", "bias", "direction_accuracy")
                    if per_seed[0].get(k) is not None},
                    "per_seed_mae": [ps["mae"] for ps in per_seed],
                    "per_seed_rmse": [ps["rmse"] for ps in per_seed]}
            result["scopes"][scope_names[scope]] = entry
        # support classes (cut-row weighted attribution): each pair's cut rows
        # share total weight 1
        for cls, name in enumerate(("T_seen", "T_unseen_frag_both_seen",
                                    "T_unseen_one_frag_unseen",
                                    "T_unseen_both_frags_unseen")):
            cut_mask = cut_class == cls
            # pair-level weight = (#cut rows of pair in class) / (#cut rows of pair)
            w = np.zeros(pd.pa.size)
            for i, cuts in enumerate(cut_lists):
                if cuts.size:
                    w[i] = float(np.sum(cut_class[cuts] == cls)) / float(cuts.size)
            sel = (w > 0) & (pd.scope == 2)
            entry = {"scope": "heldout_heldout", "n_pairs_any": int(sel.sum()),
                     "weight_mass": float(w[sel].sum())
                     if sel.any() else 0.0,
                     "n_components": int(np.unique(pd.component[sel]).size) if sel.any() else 0}
            for m, seeds_pred in data.items():
                per_seed = []
                for d in seeds_pred:
                    dp, ok = delta_of(d, pd)
                    use = sel & ok
                    per_seed.append(pair_metrics(delta_true[use], dp[use], w[use]))
                if not per_seed[0]["n"]:
                    entry[m] = {"n_scored": 0}
                    continue
                entry[m] = {k: float(np.mean([ps[k] for ps in per_seed]))
                            for k in ("mae", "rmse", "direction_accuracy")
                            if per_seed[0].get(k) is not None}
            result["support_classes"][name] = entry
        # novelty buckets on heldout x heldout (pair-level, unweighted)
        for scope in (0, 2):
            scope_mask = pd.scope == scope
            for label, values in (("order_full", eta), ("order_len3_L", np.nan),
                                  ("bag_full", eta_bag)):
                if label == "order_len3_L":
                    values = eta_len[3]
                entry = {}
                for lo, hi in zip(buckets[:-1], buckets[1:]):
                    if lo == 0.0:
                        mask = scope_mask & np.isfinite(values) & (values <= 1e-12)
                        name = "eta=0"
                    else:
                        mask = scope_mask & np.isfinite(values) & (values > lo) & (values <= hi)
                        name = f"({lo},{hi}]"
                    sub = {"n_pairs": int(mask.sum()),
                           "n_components": int(np.unique(pd.component[mask]).size)}
                    for m, seeds_pred in data.items():
                        per_seed = []
                        for d in seeds_pred:
                            dp, ok = delta_of(d, pd)
                            use = mask & ok
                            per_seed.append(pair_metrics(delta_true[use], dp[use]))
                        if per_seed and per_seed[0]["n"]:
                            sub[m] = {k: float(np.mean([ps[k] for ps in per_seed]))
                                      for k in ("mae", "rmse", "direction_accuracy")
                                      if per_seed[0].get(k) is not None}
                    entry[name] = sub
                result["novelty"].setdefault(scope_names[scope], {})[label] = entry
        # |delta| >= 0.25 sensitivity for heldout x heldout
        big = (pd.scope == 2) & (np.abs(delta_true) >= 0.25)
        sens = {"n_pairs": int(big.sum()),
                "n_components": int(np.unique(pd.component[big]).size)}
        for m, seeds_pred in data.items():
            per_seed = []
            for d in seeds_pred:
                dp, ok = delta_of(d, pd)
                use = big & ok
                per_seed.append(pair_metrics(delta_true[use], dp[use]))
            if per_seed and per_seed[0]["n"]:
                sens[m] = {"mae": float(np.mean([ps["mae"] for ps in per_seed])),
                           "rmse": float(np.mean([ps["rmse"] for ps in per_seed])),
                           "direction_accuracy": float(np.mean(
                               [ps["direction_accuracy"] for ps in per_seed
                                if ps["direction_accuracy"] is not None]))}
        result["delta_sensitivity_0p25"] = sens
        # delta == 0 removal count
        result["counts"]["direction_items_removed"] = int(
            np.sum(np.abs(delta_true[pd.scope == 2]) <= 1e-8))
        out["splits"][split_name] = result
        print(f"{split_name}: pairs {result['counts']['scope_counts']} "
              f"support {[(k, v['n_pairs_any']) for k, v in result['support_classes'].items()]}")
    C.write_json(ANALYSIS / "unique_pair_metrics.json", out)
    return out


def _dims() -> tuple[list[int], list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    return ([int(x) for x in get_atom_feature_dims()],
            [int(x) for x in get_bond_feature_dims()])


# ---------------------------------------------------------------------------
# pair residual decomposition (§9.5)
# ---------------------------------------------------------------------------


def cmd_decomp(bundle: D.DataBundle) -> dict[str, Any]:
    lock = C.read_json(ANALYSIS / "selection_lock.json")
    models = [lock["champion"], lock["M_star"], lock["ACRA_B_star"]]
    if lock.get("DIAG_twin"):
        models.append(lock["DIAG_twin"])
    models = [m for m in dict.fromkeys(models) if m]
    split = C.split_spec("D0", bundle.raw)
    pd = PairData(bundle, split)
    mask = pd.scope == 2  # heldout x heldout
    # component weights for the pair graph
    w = np.zeros(pd.pa.size)
    for c in np.unique(pd.component):
        idx = np.nonzero(pd.component == c)[0]
        w[idx] = 1.0 / (len(np.unique(pd.component)) * idx.size)
    out: dict[str, Any] = {"generated_utc": C.now_utc(),
                           "scope": "D0 heldout x heldout unique pairs",
                           "n_pairs": int(mask.sum()), "models": {}}
    y = bundle.labels
    for m in models:
        preds = []
        for seed in (0, 1, 2):
            rid = run_id(m, "D0", seed)
            path = RESULTS / rid / "predictions_best_eval.npz"
            if not path.is_file():
                path = RESULTS / rid / "predictions_final_eval.npz"
            if not path.is_file():
                continue
            with np.load(path, allow_pickle=False) as z:
                preds.append(dict(zip(np.asarray(z["row_id"]).tolist(),
                                      np.asarray(z["pred"], np.float64).tolist())))
        if not preds:
            continue
        entries = []
        for d in preds:
            e = np.full(y.size, np.nan)
            for r, v in d.items():
                e[int(r)] = v
            e = e - y
            ea, eb = e[pd.pa[mask]], e[pd.pb[mask]]
            ww = w[mask] / w[mask].sum()
            entries.append({
                "mse_delta": float(np.sum(ww * (eb - ea) ** 2)),
                "mean_ea2": float(np.sum(ww * ea ** 2)),
                "mean_eb2": float(np.sum(ww * eb ** 2)),
                "cross": float(np.sum(ww * ea * eb)),
                "corr_endpoint_residuals": float(np.corrcoef(ea, eb)[0, 1])
                if np.std(ea) > 0 and np.std(eb) > 0 else None,
            })
        keys = [k for k in entries[0] if all(x[k] is not None for x in entries)]
        out["models"][m] = {
            "mean": {k: float(np.mean([x[k] for x in entries])) for k in keys},
            "per_seed": entries,
        }
        for x in entries:
            if x["corr_endpoint_residuals"] is not None:
                out["models"][m].setdefault("corr", []).append(x["corr_endpoint_residuals"])
        out["models"][m]["check_identity"] = max(
            abs(x["mse_delta"] - (x["mean_ea2"] + x["mean_eb2"] - 2 * x["cross"]))
            for x in entries)
    C.write_json(ANALYSIS / "pair_residual_decomposition.json", out)
    print(json.dumps({m: {k: round(v, 4) for k, v in out["models"][m]["mean"].items()}
                      for m in out["models"]}, indent=1))
    return out


# ---------------------------------------------------------------------------
# molecule-level novelty strata (§9.4 / Q5)
# ---------------------------------------------------------------------------


def cmd_molnov(bundle: D.DataBundle) -> dict[str, Any]:
    lock = C.read_json(ANALYSIS / "selection_lock.json")
    models = [lock["champion"], lock["M_star"], lock["ACRA_B_star"], "GINE_REF",
              "PATH_COUNT_RIDGE", "MORGAN_COUNT_KRR", "COUNT_ONLY_RIDGE"]
    models = [m for m in dict.fromkeys(models) if m]
    cfg = C.load_config()
    out: dict[str, Any] = {"generated_utc": C.now_utc(), "splits": {}}
    buckets = [(0.0, 0.0), (0.0, 0.01), (0.01, 0.05), (0.05, 1.0 + 1e-9)]
    for split_name in ("D0", "D1"):
        f1 = None
        if split_name == "D1":
            f1 = json.loads(C.SPLIT_F1.read_text(encoding="utf-8"))
            f1["file_sha256"] = C.sha256_file(C.SPLIT_F1)
        split = C.split_spec(split_name, bundle.raw, f1=f1)
        rows_eval = np.asarray(split["eval_rows"], np.int64)
        cache_rows = np.union1d(bundle.official_train, bundle.official_valid)
        meta = PP.ensure_cache(bundle.raw, cache_rows, int(cfg["paths"]["max_len"]))
        cache = PP.PathCache(meta, device="cpu")
        vocab = set(D.fit_path_vocab(cache, split["fit_rows"], cache.L).ids)
        counts_eval = dict(zip(rows_eval.tolist(), path_count_dicts(cache, rows_eval)))
        counts_fit = dict(zip(np.asarray(split["fit_rows"]).tolist(),
                              path_count_dicts(cache, split["fit_rows"])))
        bag_of = bag_keys(cache)
        vocab_bag = {bag_of[p] for p in vocab}
        eta = np.zeros(rows_eval.size)
        eta_bag = np.zeros(rows_eval.size)
        for i, r in enumerate(rows_eval.tolist()):
            d = counts_eval[r]
            tot = sum(d.values()) or 1
            eta[i] = sum(m for k, m in d.items() if k not in vocab) / tot
            agg: dict[bytes, int] = defaultdict(int)
            for k, m in d.items():
                agg[bag_of[k]] += m
            eta_bag[i] = sum(m for k, m in agg.items() if k not in vocab_bag) / tot
        payload = C.load_split_f0() if split_name == "D0" else f1
        scaf_map = payload["scaffold_by_row"]
        fit_scaf = {str(scaf_map[str(int(r))]) for r in np.asarray(split["fit_rows"]).tolist()}
        scaf = np.asarray([str(scaf_map[str(int(r))]) for r in rows_eval.tolist()])
        scaf_novel = np.asarray([x not in fit_scaf for x in scaf], bool)
        y = bundle.labels[rows_eval]
        entry: dict[str, Any] = {
            "n_eval": int(rows_eval.size),
            "n_scaffold_novel": int(scaf_novel.sum()),
            "unseen_path_mass": {"mean": float(eta.mean()), "median": float(np.median(eta)),
                                 "share_gt0": float(np.mean(eta > 0)),
                                 "share_gt5pct": float(np.mean(eta > 0.05)),
                                 "max": float(eta.max())},
            "buckets": {}, "scaffold_cross": {},
        }
        for lo, hi in buckets:
            if lo == 0.0 and hi == 0.0:
                mask = eta <= 1e-12
                name = "eta_mol=0"
            else:
                mask = (eta > lo) & (eta <= hi)
                name = f"({lo},{hi}]"
            sub = {"n": int(mask.sum()), "scaffold_novel": int((mask & scaf_novel).sum())}
            for m in models:
                d = prediction_dict(m, split_name, 0)
                if not d:
                    continue
                pred = np.asarray([d[int(r)] for r in rows_eval.tolist()])
                if mask.sum():
                    sub[m] = {"rmse": C.rmse(pred[mask], y[mask]),
                              "mae": C.mae(pred[mask], y[mask])}
            entry["buckets"][name] = sub
        for flag, name in ((False, "scaffold_seen"), (True, "scaffold_novel")):
            mask = scaf_novel == flag
            sub = {"n": int(mask.sum()), "mean_eta": float(eta[mask].mean()) if mask.any() else 0.0}
            for m in models:
                d = prediction_dict(m, split_name, 0)
                if not d:
                    continue
                pred = np.asarray([d[int(r)] for r in rows_eval.tolist()])
                if mask.sum():
                    sub[m] = {"rmse": C.rmse(pred[mask], y[mask]),
                              "mae": C.mae(pred[mask], y[mask])}
            entry["scaffold_cross"][name] = sub
        out["splits"][split_name] = entry
        print(f"{split_name}: eta mean={eta.mean():.4f} share>0={np.mean(eta>0):.3f} "
              f"scaffold_novel={int(scaf_novel.sum())}/{rows_eval.size}")
    C.write_json(ANALYSIS / "molecule_novelty.json", out)
    return out


# ---------------------------------------------------------------------------
# size strata and worst errors (§9.7)
# ---------------------------------------------------------------------------


def cmd_sizes(bundle: D.DataBundle) -> dict[str, Any]:
    lock = C.read_json(ANALYSIS / "selection_lock.json")
    models = [lock["champion"], lock["ACRA_B_star"], "GINE_REF", "COUNT_ONLY_RIDGE"]
    models = [m for m in dict.fromkeys(models) if m]
    split = C.split_spec("D0", bundle.raw)
    rows = np.asarray(split["eval_rows"], np.int64)
    atoms_fit = np.asarray(bundle.raw["graph_n_atoms"], np.float64)[split["fit_rows"]]
    qs = np.quantile(atoms_fit, [0.25, 0.5, 0.75])
    n_atoms = np.asarray(bundle.raw["graph_n_atoms"], np.float64)[rows]
    quart = np.searchsorted(qs, n_atoms, side="right")
    y = bundle.labels[rows]
    out: dict[str, Any] = {"generated_utc": C.now_utc(), "quartile_edges": qs.tolist(),
                           "models": {}}
    per_model_err: dict[str, np.ndarray] = {}
    for m in models:
        if m in ARM_SPEC:
            rid = run_id(m, "D0", 0)
            path = RESULTS / rid / "predictions_best_eval.npz"
        else:
            path = CPU_OUT / f"CPU_{m}_D0" / "predictions.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            pred = np.asarray(z["pred"], np.float64)
        err = pred - y
        per_model_err[m] = err
        entry = {"overall_rmse": C.rmse(pred, y), "overall_mae": C.mae(pred, y)}
        for q in range(4):
            mask = quart == q
            entry[f"q{q+1}"] = {"n": int(mask.sum()),
                                "ranges": [float(qs[q - 1]) if q > 0 else 0.0,
                                           float(qs[q]) if q < 3 else float(n_atoms.max())],
                                "rmse": C.rmse(pred[mask], y[mask]),
                                "mae": C.mae(pred[mask], y[mask])}
        # top-5 squared-error union removal sensitivity (not used for selection)
        order = np.argsort(-(err ** 2))
        keep = np.ones(rows.size, bool)
        keep[order[:5]] = False
        entry["top5_removed_rmse"] = C.rmse(pred[keep], y[keep])
        entry["worst"] = [{"row_id": int(bundle.row_id[rows[i]]),
                           "atoms": float(n_atoms[i]), "true": float(y[i]),
                           "pred": float(pred[i])} for i in order[:10]]
        out["models"][m] = entry
    # union of each model's top-5 squared errors, removed post hoc (sensitivity only)
    union = np.zeros(rows.size, np.int64)
    for err in per_model_err.values():
        union[np.argsort(-(err ** 2))[:5]] += 1
    keep = union == 0
    union_entry = {"n_removed": int((~keep).sum()), "n_eval": int(keep.sum()),
                   "models": {}}
    for m in models:
        if m in ARM_SPEC:
            path = RESULTS / run_id(m, "D0", 0) / "predictions_best_eval.npz"
        else:
            path = CPU_OUT / f"CPU_{m}_D0" / "predictions.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            pred = np.asarray(z["pred"], np.float64)
        union_entry["models"][m] = {"rmse": C.rmse(pred[keep], y[keep])}
    out["top5_union_removed"] = union_entry
    C.write_json(ANALYSIS / "size_strata.json", out)
    print(json.dumps({m: {"rmse": round(out["models"][m]["overall_rmse"], 4),
                          "quartiles": [round(out["models"][m][f"q{i}"]["rmse"], 4)
                                        for i in range(1, 5)]}
                      for m in out["models"]}, indent=1))
    return out


def _pred_of(m: str, rows: np.ndarray, bundle: D.DataBundle) -> np.ndarray:
    path = RESULTS / run_id(m, "D0", 0) / "predictions_best_eval.npz"
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["pred"], np.float64)


# ---------------------------------------------------------------------------
# SUM ledger and bridge decomposition (§9.6)
# ---------------------------------------------------------------------------


def enumerate_row(cache: PP.PathCache, raw: Mapping[str, Any], row: int, L: int):
    """All geometric paths of one row with vertex tuples and canonical tokens."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    n = a1 - a0
    atom = np.asarray(raw["atom_feat"])[a0:a1]
    bond = np.asarray(raw["bond_feat"])[b0:b1]
    adj = PP.adjacency(n, np.asarray(raw["bond_u"])[b0:b1], np.asarray(raw["bond_v"])[b0:b1])
    bond_row = np.full((n, n), -1, np.int64)
    for e, (u, v) in enumerate(zip(np.asarray(raw["bond_u"])[b0:b1],
                                   np.asarray(raw["bond_v"])[b0:b1])):
        bond_row[int(u), int(v)] = e
        bond_row[int(v), int(u)] = e
    paths, _ = PP.enumerate_paths(adj, L)
    atoms = np.zeros((len(paths), L + 1, C.ATOM_COLUMNS), np.uint8)
    bonds = np.zeros((len(paths), L, C.BOND_COLUMNS), np.uint8)
    keys = []
    for i, seq in enumerate(paths):
        a, b, k = PP.tokens_for(seq, atom, bond, bond_row)
        atoms[i, :a.shape[0]] = a
        bonds[i, :b.shape[0]] = b
        keys.append(k)
    return paths, atoms, bonds, keys


def score_paths(model: M.PSCModel, atoms: np.ndarray, bonds: np.ndarray, L: int,
                device: str, bag: bool) -> np.ndarray:
    import torch

    model.eval()
    plen = np.array([int(np.max(np.nonzero(atoms[i].any(1))[0])) if atoms[i].any() else 0
                     for i in range(atoms.shape[0])], np.int64)
    # length is recoverable from the padded bond block / atom block; use the
    # canonical pad convention (all-zero rows) explicitly
    plen = np.zeros(atoms.shape[0], np.int64)
    for i in range(atoms.shape[0]):
        nz = np.nonzero(atoms[i].any(1))[0]
        plen[i] = int(nz.max()) if nz.size else 0
    scores = np.zeros(atoms.shape[0])
    a_t = torch.as_tensor(atoms).long().to(device)
    b_t = torch.as_tensor(bonds).long().to(device)
    p_t = torch.as_tensor(plen).long().to(device)
    with torch.no_grad():
        for start in range(0, atoms.shape[0], 65536):
            sl = slice(start, start + 65536)
            e_f = model.encoder(a_t[sl], b_t[sl], p_t[sl], reverse=False)
            e_r = e_f if bag else model.encoder(a_t[sl], b_t[sl], p_t[sl], reverse=True)
            scores[sl] = (0.5 * (model.head(e_f) + model.head(e_r))).double().cpu().numpy()
    return scores


def load_psc_model(rid: str, bundle: D.DataBundle, cfg: Mapping[str, Any],
                   device: str = "cpu"):
    import torch

    spec = ARM_SPEC[[a for a in ARM_SPEC if rid.startswith(a)][0]]
    man = load_manifest(rid)
    cache_meta = PP.ensure_cache(bundle.raw, np.union1d(bundle.official_train,
                                                        bundle.official_valid),
                                 int(cfg["paths"]["max_len"]))
    cache = PP.PathCache(cache_meta, device=device)
    atom_dims, bond_dims = _dims()
    model, _ = M.build_model(atom_dims, bond_dims, cache.L, spec["readout"], 0, cfg)
    state = torch.load(RESULTS / rid / "state_best.pt", map_location=device,
                       weights_only=False)
    model.load_state_dict(state["state"])
    model.eval()
    return model, cache, spec, state


def cmd_ledger(bundle: D.DataBundle) -> dict[str, Any]:
    lock = C.read_json(ANALYSIS / "selection_lock.json")
    cfg = C.load_config()
    champ = lock["champion"]
    target = champ if ARM_SPEC[champ]["readout"] == "SUM" else lock["R_star"]
    rid = run_id(target, "D0", 0)
    model, cache, spec, state = load_psc_model(rid, bundle, cfg)
    c_l = np.asarray(state["c_l"], np.float64)
    scaler = state["scaler"]
    L = cache.L
    split = C.split_spec("D0", bundle.raw)
    pd = PairData(bundle, split)
    import psc_train as T
    pred_eval = {}
    for rid2 in [run_id(lock["ACRA_B_star"], "D0", s) for s in (0, 1, 2)]:
        path = RESULTS / rid2 / "predictions_best_eval.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as z:
                pred_eval[rid2] = dict(zip(np.asarray(z["row_id"]).tolist(),
                                           np.asarray(z["pred"], np.float64).tolist()))
    # verify the ledger identity and build bridge decomposition for heldout pairs
    held = np.asarray(split["eval_rows"], np.int64)
    pairs_mask = pd.scope == 2
    delta_true = bundle.labels[pd.pb] - bundle.labels[pd.pa]
    out: dict[str, Any] = {"generated_utc": C.now_utc(), "ledger_model": target,
                           "champion": champ,
                           "note": ("SUM ledger of the champion" if target == champ else
                                    "SUM ledger reported for the matched SUM twin of a POOL "
                                    "champion; it does not explain the champion's predictions"),
                           "identity_max_abs_std": None, "pairs": 0, "examples": []}
    # identity check on 12 heldout molecules
    check_rows = held[:12]
    max_err = 0.0
    for row in check_rows.tolist():
        paths, atoms, bonds, keys = enumerate_row(cache, bundle.raw, row, L)
        s = score_paths(model, atoms, bonds, L, "cpu", spec["input"] == "BAG")
        plen = np.array([len(p) - 1 for p in paths])
        f_ledger = float(model.beta) + float(np.sum(c_l[plen] * s))
        f_pred = _predict_one(model, cache, c_l, int(row), spec["input"] == "BAG")
        max_err = max(max_err, abs(f_ledger - f_pred))
    out["identity_max_abs_std"] = float(max_err)
    if max_err > 1e-3:
        raise AssertionError(f"SUM ledger identity failed: {max_err}")
    # candidates for the predefined case selection
    cand = []
    for i in np.nonzero(pairs_mask)[0]:
        a, b = int(pd.pa[i]), int(pd.pb[i])
        dp = np.mean([np.mean([
            pred_eval[run_id(lock["ACRA_B_star"], "D0", s)][b]
            - pred_eval[run_id(lock["ACRA_B_star"], "D0", s)][a]
            for s in (0, 1, 2) if run_id(lock["ACRA_B_star"], "D0", s) in pred_eval])])
        cand.append({"pair": i, "a": a, "b": b,
                     "delta_true": float(delta_true[i]),
                     "delta_ref_pred": float(dp),
                     "abs_err_ref": float(abs(dp - delta_true[i]))})
    cand = [c for c in cand if np.isfinite(c["abs_err_ref"])]
    out["pairs"] = len(cand)
    if not cand:
        raise SystemExit("ledger: no finite reference predictions (ACRA_B* missing)")
    # exact T support for the case rules
    import psc_train as TT
    dir_ = TT.ensure_pair_directory(bundle)
    fit_T = set(np.unique(dir_.T_id[pd.cut_scope == 0]).tolist())
    trc = load_trc()
    frag_seen = fit_fragment_set(dir_, split["fit_rows"], trc)
    for entry in cand:
        cut_rows = pd.cut_rows_of_pair()[entry["pair"]]
        entry["T_seen"] = bool(np.any([int(dir_.T_id[c]) in fit_T for c in cut_rows.tolist()]))
        entry["frag_seen"] = bool(all(int(dir_.word_a[c]) in frag_seen
                                      and int(dir_.word_b[c]) in frag_seen
                                      for c in cut_rows.tolist()))
    # eta
    cache_rows = np.union1d(bundle.official_train, bundle.official_valid)
    counts = dict(zip(cache_rows.tolist(), path_count_dicts(cache, cache_rows)))
    vocab_fit = set(D.fit_path_vocab(cache, split["fit_rows"], L).ids)
    for entry in cand:
        da, db = counts[entry["a"]], counts[entry["b"]]
        keys = set(da) | set(db)
        num = sum(abs(db.get(k, 0) - da.get(k, 0)) for k in keys if k not in vocab_fit)
        den = sum(abs(db.get(k, 0) - da.get(k, 0)) for k in keys)
        entry["eta"] = num / den if den > 0 else None
    high_eta = np.quantile([e["eta"] for e in cand if e["eta"] is not None], 0.75)
    rules = cfg["analysis"]["case_rules"]
    picked: list[dict[str, Any]] = []
    seen_pairs: set[int] = set()

    def take(filter_fn, key, n, why):
        items = [e for e in cand if e["pair"] not in seen_pairs and filter_fn(e)]
        items.sort(key=key)
        for e in items[:n]:
            seen_pairs.add(e["pair"])
            picked.append({**e, "rule": why})

    unseen_frag_seen = lambda e: (not e["T_seen"]) and e["frag_seen"]
    take(unseen_frag_seen, lambda e: e["abs_err_ref"], rules["t_unseen_frag_seen_improved"],
         "T unseen / fragments seen: largest ACRA_B* error (room to improve)")
    take(lambda e: unseen_frag_seen(e) and e["abs_err_ref"] > 0, lambda e: -e["abs_err_ref"],
         rules["t_unseen_frag_seen_degraded"],
         "T unseen / fragments seen: smallest ACRA_B* error")
    take(lambda e: e["eta"] is not None and e["eta"] >= high_eta,
         lambda e: e["abs_err_ref"], rules["high_eta_improved"],
         "high eta: largest ACRA_B* error")
    take(lambda e: e["eta"] is not None and e["eta"] >= high_eta,
         lambda e: -e["abs_err_ref"], rules["high_eta_degraded"],
         "high eta: smallest ACRA_B* error")
    take(lambda e: e["T_seen"] and e["delta_true"] > 0, lambda e: -e["delta_true"],
         rules["seen_t_positive"], "seen T with positive true delta")
    take(lambda e: e["T_seen"] and e["delta_true"] < 0, lambda e: e["delta_true"],
         rules["seen_t_negative"], "seen T with negative true delta")
    # bridge decomposition for the picked pairs
    parse_mol = np.asarray(trc["parse_mol"], np.int64)
    parse_S = np.asarray(trc["parse_S"], np.int64)
    parse_C = np.asarray(trc["parse_C"], np.int64)
    parse_bond = np.asarray(trc["parse_cut_bond_row"], np.int64)
    parse_kind = np.asarray(trc["parse_kind"], np.uint8)
    parse_S_atoms = np.asarray(trc["parse_S_atoms"], np.int64)
    parse_S_ptr = np.asarray(trc["parse_S_atoms_ptr"], np.int64)
    parse_C_atoms = np.asarray(trc["parse_C_atoms"], np.int64)
    parse_C_ptr = np.asarray(trc["parse_C_atoms_ptr"], np.int64)

    def parse_of(mol, word, ctx, bond):
        m = (parse_mol == mol) & (parse_S == word) & (parse_C == ctx) \
            & (parse_bond == bond) & (parse_kind == 0)
        idx = np.nonzero(m)[0]
        return int(idx[0]) if idx.size else None

    for entry in picked:
        results = {}
        for side, row in (("a", entry["a"]), ("b", entry["b"])):
            paths, atoms, bonds, keys = enumerate_row(cache, bundle.raw, row, L)
            s = score_paths(model, atoms, bonds, L, "cpu", spec["input"] == "BAG")
            plen = np.array([len(p) - 1 for p in paths])
            results[side] = {"paths": paths, "scores": s, "plen": plen, "row": row}
        # use the first cut row of the pair for the S/C atom sets
        cut_rows = pd.cut_rows_of_pair()[entry["pair"]]
        c0 = int(cut_rows[0])
        ctx = int(trc["pair_C"][c0])
        p_a = parse_of(entry["a"], int(pd.dir.word_a[c0]), ctx, int(pd.dir.bond_id[c0]))
        p_b = parse_of(entry["b"], int(pd.dir.word_b[c0]), ctx, int(pd.dir.bond_id[c0]))
        if p_a is None or p_b is None:
            entry["decomposition"] = None
            continue
        S_a = set(parse_S_atoms[parse_S_ptr[p_a]:parse_S_ptr[p_a + 1]].tolist())
        C_a = set(parse_C_atoms[parse_C_ptr[p_a]:parse_C_ptr[p_a + 1]].tolist())
        S_b = set(parse_S_atoms[parse_S_ptr[p_b]:parse_S_ptr[p_b + 1]].tolist())
        C_b = set(parse_C_atoms[parse_C_ptr[p_b]:parse_C_ptr[p_b + 1]].tolist())

        def classed(side, S, C):
            sums = {"S": 0.0, "C": 0.0, "X": 0.0}
            for seq, sc, l in zip(results[side]["paths"], results[side]["scores"],
                                  results[side]["plen"]):
                vs = set(int(x) for x in seq)
                w = c_l[l]
                if vs <= S:
                    sums["S"] += w * sc
                elif vs <= C:
                    sums["C"] += w * sc
                else:
                    sums["X"] += w * sc
            return sums
        sa = classed("a", S_a, C_a)
        sb = classed("b", S_b, C_b)
        entry["decomposition"] = {
            "F_a": float(model.beta) + sa["S"] + sa["C"] + sa["X"],
            "F_b": float(model.beta) + sb["S"] + sb["C"] + sb["X"],
            "delta_F": (sb["S"] - sa["S"]) + (sb["C"] - sa["C"]) + (sb["X"] - sa["X"]),
            "F0_S": {"a": sa["S"], "b": sb["S"]},
            "F0_C": {"a": sa["C"], "b": sb["C"]},
            "X": {"a": sa["X"], "b": sb["X"]},
            "delta_F0_S": sb["S"] - sa["S"], "delta_F0_C": sb["C"] - sa["C"],
            "delta_X": sb["X"] - sa["X"],
            "delta_true": entry["delta_true"],
            "delta_pred_scaled": (sb["S"] - sa["S"] + sb["C"] - sa["C"]
                                  + sb["X"] - sa["X"]) * scaler["std"],
        }
        out["examples"].append(entry)
    C.write_json(ANALYSIS / "sum_edit_ledger.json", out)
    print(f"ledger: model={target} identity_err={max_err:.2e} examples={len(out['examples'])}")
    return out


def _predict_one(model, cache, c_l, row, bag) -> float:
    """Standardised-space prediction of the SUM/Pool readout for one molecule."""
    import psc_train as T

    pred, _ = T.predict_psc(model, cache, c_l, [int(row)], "cpu", bag, 1)
    return float(pred[0])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def cmd_figures(bundle: D.DataBundle) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lock = C.read_json(ANALYSIS / "selection_lock.json")
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    agg = summary["aggregate"]["D0"]
    fig_dir = ANALYSIS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # 1. D0 performance across seeds
    arms = [a for a in ("PSC_SUM", "PSC_SUM_PAIR", "PSC_SUM_BAG", "PSC_SUM_BAG_PAIR",
                        "PSC_POOL", "PSC_POOL_PAIR", "PSC_POOL_BAG", "PSC_POOL_BAG_PAIR",
                        "ACRA_REF", "ACRA_PAIR", "GINE_REF")
            if a in agg]
    cpu = {m: agg.get(m) for m in CPU_MODELS if m in agg}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(arms))
    means = [agg[a]["rmse_mean"] for a in arms]
    sds = [agg[a]["rmse_sd"] for a in arms]
    ax.bar(xs, means, yerr=sds, color="#4c72b0", alpha=0.85, capsize=3)
    for a, x in zip(arms, xs):
        ax.scatter([x] * len(agg[a]["rmse_seeds"]), agg[a]["rmse_seeds"], s=10, color="k")
    for name, v in cpu.items():
        ax.axhline(v["rmse_mean"], ls="--", lw=1, label=name)
    for name, ls in (("ACRA_PAIR", ":"),):
        if name in agg:
            ax.axhline(agg[name]["rmse_mean"], ls=ls, lw=1, color="crimson",
                       label="ACRA_B* baseline")
    ax.set_xticks(xs)
    ax.set_xticklabels([a.replace("PSC_", "") for a in arms], rotation=45, ha="right")
    ax.set_ylabel("whole-molecule RMSE (official label units)")
    ax.set_title("D0 (Train-sub 2856 / Dev 504): seed mean ± SD; dots are seeds")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_d0_performance.png", dpi=160)
    plt.close(fig)
    # 2. matched mechanism contrasts
    pairs = [("PSC_SUM", "PSC_SUM_BAG"), ("PSC_POOL", "PSC_POOL_BAG"),
             ("PSC_SUM", "PSC_POOL"), ("PSC_SUM_BAG", "PSC_POOL_BAG"),
             ("PSC_SUM_PAIR", "PSC_SUM_BAG_PAIR"), ("PSC_POOL_PAIR", "PSC_POOL_BAG_PAIR"),
             ("ACRA_REF", "ACRA_PAIR")]
    if lock.get("DIAG_twin"):
        pairs.append((lock["champion"], lock["DIAG_twin"]))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = []
    for i, (a, b) in enumerate(pairs):
        if a in agg and b in agg:
            d = np.asarray(agg[b]["rmse_seeds"]) - np.asarray(agg[a]["rmse_seeds"])
            ax.barh(i, d.mean(), xerr=d.std(ddof=1) if d.size > 1 else 0, color="#dd8452")
            ax.scatter(d, [i] * d.size, color="k", s=10)
            labels.append(f"{a.replace('PSC_','')} → {b.replace('PSC_','')}")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Δ RMSE (target arm − reference arm); negative = target better")
    ax.set_title("One-factor contrasts on D0, paired by seed")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_contrasts.png", dpi=160)
    plt.close(fig)
    # 3. novelty / support strata
    if (ANALYSIS / "unique_pair_metrics.json").is_file():
        upm = json.loads((ANALYSIS / "unique_pair_metrics.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key in zip(axes, ("support_classes", "novelty")):
            d = upm["splits"]["D0"]
            if key == "support_classes":
                names = list(d["support_classes"])
                vals = {}
                for m in [lock["champion"], lock["ACRA_B_star"]]:
                    vals[m] = [d["support_classes"][n].get(m, {}).get("mae") for n in names]
                x = np.arange(len(names))
                for j, m in enumerate(vals):
                    ax.bar(x + 0.35 * j, vals[m], width=0.35, label=m)
                ax.set_xticks(x)
                ax.set_xticklabels(["T seen", "T unseen\nfrags seen", "1 frag unseen",
                                    "2 frags unseen"], fontsize=7)
                ax.set_ylabel("pair Δ MAE (cut-row weighted)")
                ax.set_title("Support classes, D0 heldout pairs")
                ax.legend(fontsize=7)
            else:
                names = list(d["novelty"]["heldout_heldout"]["order_full"])
                x = np.arange(len(names))
                for j, m in enumerate([lock["champion"], lock["ACRA_B_star"]]):
                    vals = [d["novelty"]["heldout_heldout"]["order_full"][n].get(m, {})
                            .get("mae") for n in names]
                    ax.bar(x + 0.35 * j, vals, width=0.35, label=m)
                ax.set_xticks(x)
                ax.set_xticklabels(names, fontsize=7)
                ax.set_ylabel("pair Δ MAE")
                ax.set_title("ORDER path-word novelty, D0 heldout × heldout")
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig3_novelty_support.png", dpi=160)
        plt.close(fig)
    # 4. training curves of the champion family
    fig, ax = plt.subplots(figsize=(7, 4))
    for arm, color in ((lock["champion"], "#4c72b0"), (lock["ACRA_B_star"], "#c44e52"),
                       ("GINE_REF", "#55a868")):
        rid = run_id(arm, "D0", 0)
        p = RESULTS / rid / "curves.csv"
        if not p.is_file():
            continue
        rows = C.read_csv(p) if hasattr(C, "read_csv") else None
        if rows is None:
            import csv
            with p.open() as fh:
                rows = list(csv.DictReader(fh))
        epochs = [int(r["epoch"]) for r in rows if r.get("eval_rmse")]
        rmse = [float(r["eval_rmse"]) for r in rows if r.get("eval_rmse")]
        ax.plot(epochs, rmse, label=f"{arm} (D0 seed 0)", color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel("D0 dev RMSE")
    ax.set_title("Training curves (dev RMSE per epoch)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_curves.png", dpi=160)
    plt.close(fig)
    return {"figures": [str(p.name) for p in sorted(fig_dir.glob("*.png"))]}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("table", "select", "bootstrap", "pairs", "decomp",
                             "sizes", "molnov", "ledger", "figures", "all"))
    args = ap.parse_args(argv)
    bundle = D.load_bundle()
    stages = {"table": cmd_table, "select": cmd_select,
              "bootstrap": lambda b: cmd_bootstrap(b, "D0"),
              "pairs": cmd_pairs, "decomp": cmd_decomp, "sizes": cmd_sizes,
              "molnov": cmd_molnov, "ledger": cmd_ledger, "figures": cmd_figures}
    order = ["table", "select", "bootstrap", "pairs", "decomp", "sizes", "molnov",
             "ledger", "figures"] if args.stage == "all" else [args.stage]
    for name in order:
        print(f"== stage {name}")
        stages[name](bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())