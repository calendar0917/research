"""Analysis for Phases A / B / C: recomputed metrics and paired bootstraps.

Every number in the report is recomputed here from the stored per-molecule
predictions; nothing is copied from the run JSONs.  Bootstraps reuse the
predecessor round's paired scaffold bootstrap (1 000 resamples, seed 0, whole
Bemis–Murcko scaffold groups) so the intervals are directly comparable with
MSDS-v0.

Outputs
  analysis/metrics_molhiv.csv       arm × split AUC / AP (recomputed)
  analysis/contrasts_molhiv.json    paired scaffold bootstrap per contrast
  analysis/feature_usage.csv        block / family gain of the fitted trees
  analysis/phaseB_cross_model.json  XGB vs MLP improvement structure
  analysis/metrics_zinc.csv         ZINC MAE per arm and split
  analysis/contrasts_zinc.json      paired molecule bootstrap of MAE differences
  analysis/predictions/*.npz        consolidated per-molecule predictions
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

import sga_common as K

K.add_prev_code_path()

import msds_data as D  # noqa: E402
import msds_stats as ST  # noqa: E402

PHASE_A = ("X0", "X-MC", "X-MU", "X-WL", "X-C", "X-F", "X-PRES", "X-CF")
PHASE_B = ("N0", "N-WL", "N-CF")
CONTRASTS_A = (
    ("X-CF", "X-MC"), ("X-MU", "X-MC"), ("X-CF", "X-MU"), ("X-CF", "X-WL"),
    ("X-CF", "X-C"), ("X-CF", "X-F"), ("X-CF", "X-PRES"),
    ("X-MC", "X0"), ("X-MU", "X0"), ("X-WL", "X0"), ("X-C", "X0"), ("X-F", "X0"),
    ("X-PRES", "X0"), ("X-CF", "X0"),
)
CONTRASTS_B = (("N-CF", "N0"), ("N-WL", "N0"), ("N-CF", "N-WL"))


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_molhiv_predictions() -> dict[str, dict[str, Any]]:
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in D.load_molecules().split.items()}
    y = D.load_molecules().label.astype(np.float64)
    out: dict[str, dict[str, Any]] = {}

    hist: dict[str, np.ndarray] = {}
    for name in ("valid", "test"):
        with np.load(K.HIST_PRED_DIR / f"ogbg-molhiv_{name}_predictions.npz") as z:
            rows = np.asarray(z["rows"], dtype=np.int64)
            if not np.array_equal(rows, splits[name]):
                raise SystemExit(f"historical {name} rows mismatch")
            hist[name] = np.asarray(z["pred"]).reshape(-1).astype(np.float64)
    out["X0"] = {"kind": "reused_historical", "pred": hist}

    with np.load(K.MSDS_DATA / "phaseD_xgb_struct_predictions.npz") as z:
        out["X-CF"] = {
            "kind": "reused_MSDS_v0_phaseD",
            "pred": {n: np.asarray(z[n], dtype=np.float64) for n in ("train", "valid", "test")},
        }
    for arm in ("X-MC", "X-MU", "X-WL", "X-C", "X-F", "X-PRES"):
        with np.load(K.RUNS / "phaseA" / f"{arm}_pred.npz") as z:
            out[arm] = {"kind": "sga_v1_phaseA",
                        "pred": {n: np.asarray(z[n], dtype=np.float64) for n in ("train", "valid", "test")}}
    for arm in ("N0", "N-WL", "N-CF"):
        with np.load(K.RUNS / "phaseB" / f"{arm}_pred.npz") as z:
            out[arm] = {"kind": "sga_v1_phaseB",
                        "pred": {n: np.asarray(z[n], dtype=np.float64) for n in ("train", "valid", "test")}}
    return {"predictions": out, "splits": splits, "y": y}


def scaffold_groups() -> np.ndarray:
    molecules = D.load_molecules()
    scaffolds, _ = D.load_scaffolds(molecules)
    uniq = {s: i for i, s in enumerate(sorted(set(scaffolds)))}
    return np.asarray([uniq[s] for s in scaffolds], dtype=np.int64)


# ---------------------------------------------------------------------------
# metrics + contrasts
# ---------------------------------------------------------------------------


def main() -> None:
    K.ensure_dirs()
    (K.ANALYSIS / "predictions").mkdir(parents=True, exist_ok=True)
    data = load_molhiv_predictions()
    preds, splits, y = data["predictions"], data["splits"], data["y"]
    groups = scaffold_groups()

    # ---- recomputed metrics
    rows = []
    for arm in PHASE_A + PHASE_B:
        for split in ("train", "valid", "test"):
            if split not in preds[arm]["pred"]:
                continue
            r = splits[split]
            m = ST.auc_ap(y[r], preds[arm]["pred"][split])
            rows.append({"arm": arm, "split": split, "n": m["n"], "n_positive": m["n_positive"],
                         "roc_auc": m["roc_auc"], "average_precision": m["average_precision"],
                         "source": preds[arm]["kind"]})
    _write_csv(K.ANALYSIS / "metrics_molhiv.csv", rows)
    _write_csv(K.ANALYSIS / "metrics_phaseA.csv", [r for r in rows if r["arm"] in PHASE_A])
    _write_csv(K.ANALYSIS / "metrics_phaseB.csv", [r for r in rows if r["arm"] in PHASE_B])

    # consolidated per-molecule predictions
    for arm in PHASE_A + PHASE_B:
        payload = {"y": y}
        for split in ("train", "valid", "test"):
            if split in preds[arm]["pred"]:
                payload[split] = preds[arm]["pred"][split]
                payload[f"{split}_rows"] = splits[split]
        np.savez_compressed(K.ANALYSIS / "predictions" / f"{arm}.npz", **payload)

    # ---- paired scaffold bootstrap
    contrasts: dict[str, Any] = {"phaseA": {}, "phaseB": {}}
    for arm_a, arm_b in CONTRASTS_A:
        key = f"{arm_a}_minus_{arm_b}"
        contrasts["phaseA"][key] = {}
        for split in ("valid", "test"):
            r = splits[split]
            boot = ST.paired_scaffold_bootstrap(
                y[r], preds[arm_a]["pred"][split], preds[arm_b]["pred"][split], groups[r],
                label_a=arm_a, label_b=arm_b)
            contrasts["phaseA"][key][split] = {
                "delta_roc_auc": boot["point"]["delta_roc_auc"],
                "delta_average_precision": boot["point"]["delta_average_precision"],
                "roc_auc_a": boot["point"]["roc_auc_a"], "roc_auc_b": boot["point"]["roc_auc_b"],
                "average_precision_a": boot["point"]["average_precision_a"],
                "average_precision_b": boot["point"]["average_precision_b"],
                "bootstrap_roc_auc": boot["delta_roc_auc"],
                "bootstrap_average_precision": boot["delta_average_precision"],
                "n_bootstrap_valid": boot["n_valid"],
                "evidence_label": "EXPOSED_DEVELOPMENT_TEST" if split == "test" else "DEVELOPMENT_VALID",
            }
        va = contrasts["phaseA"][key]["valid"]["delta_roc_auc"]
        te = contrasts["phaseA"][key]["test"]["delta_roc_auc"]
        contrasts["phaseA"][key]["split_conflict"] = bool(va * te < 0)
        print(f"[A] {key:22s} valid {va:+.4f} "
              f"[{contrasts['phaseA'][key]['valid']['bootstrap_roc_auc']['p2.5']:+.4f},"
              f"{contrasts['phaseA'][key]['valid']['bootstrap_roc_auc']['p97.5']:+.4f}] | "
              f"test {te:+.4f} "
              f"[{contrasts['phaseA'][key]['test']['bootstrap_roc_auc']['p2.5']:+.4f},"
              f"{contrasts['phaseA'][key]['test']['bootstrap_roc_auc']['p97.5']:+.4f}]"
              f"{'  SPLIT_CONFLICT' if contrasts['phaseA'][key]['split_conflict'] else ''}", flush=True)
    for arm_a, arm_b in CONTRASTS_B:
        key = f"{arm_a}_minus_{arm_b}"
        contrasts["phaseB"][key] = {}
        for split in ("valid", "test"):
            r = splits[split]
            boot = ST.paired_scaffold_bootstrap(
                y[r], preds[arm_a]["pred"][split], preds[arm_b]["pred"][split], groups[r],
                label_a=arm_a, label_b=arm_b)
            contrasts["phaseB"][key][split] = {
                "delta_roc_auc": boot["point"]["delta_roc_auc"],
                "delta_average_precision": boot["point"]["delta_average_precision"],
                "roc_auc_a": boot["point"]["roc_auc_a"], "roc_auc_b": boot["point"]["roc_auc_b"],
                "average_precision_a": boot["point"]["average_precision_a"],
                "average_precision_b": boot["point"]["average_precision_b"],
                "bootstrap_roc_auc": boot["delta_roc_auc"],
                "bootstrap_average_precision": boot["delta_average_precision"],
                "evidence_label": "EXPOSED_DEVELOPMENT_TEST" if split == "test" else "DEVELOPMENT_VALID",
            }
        va = contrasts["phaseB"][key]["valid"]["delta_roc_auc"]
        te = contrasts["phaseB"][key]["test"]["delta_roc_auc"]
        contrasts["phaseB"][key]["split_conflict"] = bool(va * te < 0)
        print(f"[B] {key:22s} valid {va:+.4f} test {te:+.4f}"
              f"{'  SPLIT_CONFLICT' if contrasts['phaseB'][key]['split_conflict'] else ''}", flush=True)
    K.write_json(K.ANALYSIS / "contrasts_molhiv.json", contrasts)

    # ---- XGB vs MLP improvement structure
    K.write_json(K.ANALYSIS / "phaseB_cross_model.json", cross_model(preds, splits, y, groups))

    # ---- feature usage tables
    usage_rows, usage_detail = feature_usage()
    _write_csv(K.ANALYSIS / "feature_usage.csv", usage_rows)
    K.write_json(K.ANALYSIS / "feature_usage_detail.json", usage_detail)

    # ---- ZINC
    zinc_metrics, zinc_contrasts = zinc_analysis()
    _write_csv(K.ANALYSIS / "metrics_zinc.csv", zinc_metrics)
    K.write_json(K.ANALYSIS / "contrasts_zinc.json", zinc_contrasts)


# ---------------------------------------------------------------------------
# Phase B cross-model analysis
# ---------------------------------------------------------------------------


def _p(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))


def cross_model(preds, splits, y, groups) -> dict[str, Any]:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score

    out: dict[str, Any] = {
        "question": "do XGBoost and the MLP gain from explicit structure on the same molecules?",
        "note": "deltas are per-molecule probability differences of the paired arms; "
                "the correlation describes where the improvement sits, not its size",
    }
    for split in ("valid", "test"):
        r = splits[split]
        d_xgb = _p(preds["X-CF"]["pred"][split]) - _p(preds["X0"]["pred"][split])
        d_wl_xgb = _p(preds["X-WL"]["pred"][split]) - _p(preds["X0"]["pred"][split])
        d_mlp = _p(preds["N-CF"]["pred"][split]) - _p(preds["N0"]["pred"][split])
        d_wl_mlp = _p(preds["N-WL"]["pred"][split]) - _p(preds["N0"]["pred"][split])
        k = max(1, int(round(0.10 * r.size)))
        top_xgb = set(np.argsort(-d_xgb)[:k].tolist())
        top_mlp = set(np.argsort(-d_mlp)[:k].tolist())
        out[split] = {
            "n": int(r.size),
            "xgb_add": {"mean_delta_prob": float(d_xgb.mean()), "share_positive": float((d_xgb > 0).mean()),
                        "delta_auc": float(roc_auc_score(y[r], _p(preds["X-CF"]["pred"][split]))
                                           - roc_auc_score(y[r], _p(preds["X0"]["pred"][split])))},
            "mlp_add": {"mean_delta_prob": float(d_mlp.mean()), "share_positive": float((d_mlp > 0).mean()),
                        "delta_auc": float(roc_auc_score(y[r], _p(preds["N-CF"]["pred"][split]))
                                           - roc_auc_score(y[r], _p(preds["N0"]["pred"][split])))},
            "correlation_xgb_vs_mlp_delta": {
                "spearman": float(spearmanr(d_xgb, d_mlp).statistic),
                "pearson": float(np.corrcoef(d_xgb, d_mlp)[0, 1]),
                "sign_agreement": float(np.mean(np.sign(d_xgb) == np.sign(d_mlp))),
            },
            "top_decile_overlap": {
                "k": int(k),
                "jaccard": float(len(top_xgb & top_mlp) / max(len(top_xgb | top_mlp), 1)),
                "n_shared": int(len(top_xgb & top_mlp)),
            },
            "wl_branch": {
                "xgb_delta_auc": float(roc_auc_score(y[r], _p(preds["X-WL"]["pred"][split]))
                                       - roc_auc_score(y[r], _p(preds["X0"]["pred"][split]))),
                "mlp_delta_auc": float(roc_auc_score(y[r], _p(preds["N-WL"]["pred"][split]))
                                       - roc_auc_score(y[r], _p(preds["N0"]["pred"][split]))),
                "spearman_delta": float(spearmanr(d_wl_xgb, d_wl_mlp).statistic),
            },
            "evidence_label": "EXPOSED_DEVELOPMENT_TEST" if split == "test" else "DEVELOPMENT_VALID",
        }
    # train-fit vs transfer for each MLP arm
    out["train_fit_vs_transfer"] = {
        arm: {
            split: float(ST.auc_ap(y[splits[split]], preds[arm]["pred"][split])["roc_auc"])
            for split in ("train", "valid", "test")
        } for arm in ("N0", "N-WL", "N-CF")
    }
    return out


# ---------------------------------------------------------------------------
# feature usage
# ---------------------------------------------------------------------------


def feature_usage() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(K.MSDS_DATA / "vocab.npz") as z:
        coarse_family = np.asarray(z["coarse_family"])
        full_family = np.asarray(z["full_family"])
        coarse_ids = np.asarray(z["coarse_ids"])
        full_ids = np.asarray(z["full_ids"])
    families = ("size-2", "size-3", "size-4", "cycle-5", "cycle-6")
    # family label of a (block, block_column) pair
    def fam_of(block: str, col: int) -> str:
        if block == "C":
            return families[int(coarse_family[col])]
        if block == "F":
            return families[int(full_family[col])]
        if block == "PRES":
            return families[int(coarse_family[col])] if col < 754 else families[int(full_family[col - 754])]
        return "n/a"

    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for arm, block in (("X-C", "C"), ("X-F", "F"), ("X-PRES", "PRES")):
        payload = json.loads((K.RUNS / "phaseA" / f"{arm}.json").read_text())
        usage = payload["feature_usage"]
        block_meta = usage["blocks"][block]
        offset = block_meta["offset"]
        size = block_meta["size"]
        per_family: dict[str, dict[str, float]] = {f: {"n_used": 0, "gain": 0.0} for f in families}
        for feat in usage["top_structural_features"]:
            col = feat["block_column"]
            f = fam_of(block, col)
            per_family[f]["n_used"] += 1
            per_family[f]["gain"] += feat["total_gain"]
        total_struct = max(usage["structural_block_gain"], 1e-12)
        for f in families:
            rows.append({
                "arm": arm, "block": block, "family": f,
                "n_columns": int(size), "n_used_top60": per_family[f]["n_used"],
                "gain_top60": per_family[f]["gain"],
                "gain_share_top60": per_family[f]["gain"] / total_struct,
            })
        rows.append({
            "arm": arm, "block": block, "family": "ALL",
            "n_columns": int(size), "n_used_top60": usage["n_used_features_structural"],
            "gain_top60": usage["structural_block_gain"],
            "gain_share_top60": usage["structural_block_gain_share"],
        })
        detail[arm] = {
            "n_columns_total": usage["n_columns_total"],
            "historical_block_gain": usage["historical_block_gain"],
            "structural_block_gain": usage["structural_block_gain"],
            "structural_block_gain_share": usage["structural_block_gain_share"],
            "n_used_features_structural": usage["n_used_features_structural"],
            "top_features": [
                {**feat, "family": fam_of(block, feat["block_column"])}
                for feat in usage["top_structural_features"][:25]
            ],
        }
    # every arm that was fitted in this round gets a detail entry, so the numbers in
    # REPORT.md table 4.1 stay reproducible from committed files alone
    for arm in ("X-MC", "X-MU", "X-WL"):
        usage = json.loads((K.RUNS / "phaseA" / f"{arm}.json").read_text())["feature_usage"]
        block = usage["blocks"] and next(iter(usage["blocks"]))
        detail[arm] = {
            "n_columns_total": usage["n_columns_total"],
            "n_used_features_structural": usage["n_used_features_structural"],
            "n_columns_structural": int(usage["blocks"][block]["size"]),
            "structural_block": block,
            "structural_block_gain": usage["structural_block_gain"],
            "structural_block_gain_share": usage["structural_block_gain_share"],
            "historical_block_gain": usage["historical_block_gain"],
            "top_features": usage["top_structural_features"][:20],
        }
    return rows, detail


# ---------------------------------------------------------------------------
# ZINC
# ---------------------------------------------------------------------------


def zinc_analysis() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(K.CACHE / "zinc_design.npz") as z:
        y = np.asarray(z["y"], dtype=np.float64)
        rows = {k: np.asarray(z[f"rows_{k}"], dtype=np.int64) for k in ("train", "valid", "test")}
    preds = {}
    for arm in ("ZX0", "ZX1", "ZN0", "ZN1"):
        with np.load(K.RUNS / "phaseC" / f"{arm}_pred.npz") as z:
            preds[arm] = {k: np.asarray(z[k], dtype=np.float64) for k in ("train", "valid", "test")}
    metrics = []
    for arm, p in preds.items():
        for split in ("train", "valid", "test"):
            e = y[rows[split]] - p[split]
            metrics.append({
                "arm": arm, "split": split, "n": int(e.size),
                "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))),
                "median_ae": float(np.median(np.abs(e))),
                "spearman": float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
                    y[rows[split]], p[split]).statistic),
            })
    contrasts: dict[str, Any] = {}
    for arm_a, arm_b in (("ZX1", "ZX0"), ("ZN1", "ZN0")):
        key = f"{arm_a}_minus_{arm_b}"
        contrasts[key] = {"definition": "positive = MAE decrease (improvement)"}
        for split in ("valid", "test"):
            contrasts[key][split] = paired_molecule_bootstrap_mae(
                y[rows[split]], preds[arm_a][split], preds[arm_b][split], label_a=arm_a, label_b=arm_b)
            contrasts[key][split]["evidence_label"] = (
                "EXPOSED_DEVELOPMENT_TEST" if split == "test" else "DEVELOPMENT_VALID")
    contrasts["cross_readout"] = {
        "xgb_gain_valid": -contrasts["ZX1_minus_ZX0"]["valid"]["delta_mae"],
        "xgb_gain_test": -contrasts["ZX1_minus_ZX0"]["test"]["delta_mae"],
        "mlp_gain_valid": -contrasts["ZN1_minus_ZN0"]["valid"]["delta_mae"],
        "mlp_gain_test": -contrasts["ZN1_minus_ZN0"]["test"]["delta_mae"],
        "note": "positive = lower MAE. No cross-training-seed stability evidence exists; "
                "each arm is a single seed-0 fit.",
    }
    return metrics, contrasts


def paired_molecule_bootstrap_mae(
    y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, *, n_boot: int = 1000,
    seed: int = K.SEED, label_a: str = "a", label_b: str = "b",
) -> dict[str, Any]:
    """Paired molecule bootstrap of ``MAE_b - MAE_a`` (positive = a is better)."""
    y = np.asarray(y, dtype=np.float64)
    ea = np.abs(y - np.asarray(pred_a, dtype=np.float64))
    eb = np.abs(y - np.asarray(pred_b, dtype=np.float64))
    rng = np.random.default_rng(seed)
    n = y.size
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = eb[idx].mean() - ea[idx].mean()  # same sign convention as delta_mae
    # delta_mae > 0 means A has the LOWER error (and so does its bootstrap interval)
    return {
        "contrast": f"{label_a} vs {label_b}",
        "n": int(n), "n_bootstrap": int(n_boot), "seed": int(seed),
        "mae_a": float(ea.mean()), "mae_b": float(eb.mean()),
        "delta_mae": float(eb.mean() - ea.mean()),
        "delta_mae_definition": "MAE(b) - MAE(a); positive = a has the lower error",
        "bootstrap_delta_mae": {
            "mean": float(deltas.mean()),
            "p2.5": float(np.percentile(deltas, 2.5)),
            "p97.5": float(np.percentile(deltas, 97.5)),
            "crosses_zero": bool(np.percentile(deltas, 2.5) <= 0 <= np.percentile(deltas, 97.5)),
        },
        "molecules_improved": int((ea < eb).sum()),
        "molecules_worsened": int((ea > eb).sum()),
        "note": "paired molecule bootstrap over fixed predictions; it describes sampling "
                "uncertainty of these molecules, not training-seed stability",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
