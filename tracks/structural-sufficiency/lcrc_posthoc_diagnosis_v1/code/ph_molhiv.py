"""P1-A: MolHIV structural support, train similarity and rank-error decomposition."""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ph_common as C
import ph_rank as R
import ph_structures as S

SUPPORT_BINS = (
    ("0", lambda x: x == 0.0),
    ("(0,0.25]", lambda x: 0.0 < x <= 0.25),
    ("(0.25,0.50]", lambda x: 0.25 < x <= 0.50),
    ("(0.50,1]", lambda x: 0.50 < x <= 1.0),
)
SIM_BINS = (
    ("[0,0.30)", lambda x: x < 0.30),
    ("[0.30,0.50)", lambda x: 0.30 <= x < 0.50),
    ("[0.50,0.70)", lambda x: 0.50 <= x < 0.70),
    ("[0.70,1]", lambda x: x >= 0.70),
)
SIZE_BINS = (
    ("<=15", lambda x: x <= 15),
    ("16-25", lambda x: 16 <= x <= 25),
    (">25", lambda x: x > 25),
)


def _bin(value: float, bins) -> str:
    for name, pred in bins:
        if pred(float(value)):
            return name
    return "UNASSIGNED"


def _support_ok(n_pos: int, n_neg: int) -> bool:
    return n_pos >= 10 and n_neg >= 30


def _load_units() -> tuple[dict[tuple[str, int], np.ndarray], np.ndarray, np.ndarray]:
    units = {}
    rows0 = None
    y0 = None
    for seed in C.MOLECULE_SEEDS:
        for model in C.MODELS:
            z = C.load_mol("ogbg-molhiv", model, seed)
            rows = np.asarray(z["rows"], dtype=np.int64)
            y = np.asarray(z["y_true"], dtype=np.float64).reshape(-1)
            score = np.asarray(z["pred"], dtype=np.float64).reshape(-1)
            if rows0 is None:
                rows0, y0 = rows, y
            else:
                if not np.array_equal(rows, rows0):
                    raise RuntimeError(f"MolHIV {model}/s{seed}: row mismatch")
                if not np.array_equal(y, y0):
                    raise RuntimeError(f"MolHIV {model}/s{seed}: label mismatch")
            units[(model, seed)] = score
    return units, rows0, y0


def _group_masks(features: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {"wl2_unseen": {}, "train_tanimoto": {}, "heavy_atoms": {}}
    for name, _ in SUPPORT_BINS:
        out["wl2_unseen"][name] = np.asarray(
            [name == _bin(v, SUPPORT_BINS) for v in features["fraction_wl2_unseen"]], dtype=bool
        )
    for name, _ in SIM_BINS:
        out["train_tanimoto"][name] = np.asarray(
            [name == _bin(v, SIM_BINS) for v in features["max_train_tanimoto"]], dtype=bool
        )
    for name, _ in SIZE_BINS:
        out["heavy_atoms"][name] = np.asarray(
            [name == _bin(v, SIZE_BINS) for v in features["n_heavy_atoms"]], dtype=bool
        )
    return out


def _unit_summaries(score: np.ndarray, y: np.ndarray, masks: dict[str, dict[str, np.ndarray]]):
    out = {"overall": R.rank_summary(score, y)}
    for dim, groups in masks.items():
        out[dim] = {}
        for name, mask in groups.items():
            if mask.sum() == 0 or np.unique(y[mask]).size != 2:
                out[dim][name] = None
                continue
            out[dim][name] = R.rank_summary(score[mask], y[mask])
    return out


def _bootstrap_group(
    mask: np.ndarray,
    y: np.ndarray,
    units: dict[tuple[str, int], np.ndarray],
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    pos = np.flatnonzero(mask & (y == 1))
    neg = np.flatnonzero(mask & (y == 0))
    if pos.size == 0 or neg.size == 0:
        return {"status": "UNAVAILABLE"}
    d_pos = []
    d_neg = []
    for _ in range(n_boot):
        take_pos = rng.choice(pos, size=pos.size, replace=True)
        take_neg = rng.choice(neg, size=neg.size, replace=True)
        idx = np.concatenate([take_pos, take_neg])
        per_seed_pos = []
        per_seed_neg = []
        for seed in C.MOLECULE_SEEDS:
            a0 = units[("A0", seed)][idx]
            c1 = units[("C1", seed)][idx]
            yy = y[idx]
            per_seed_pos.append(
                float(R.positive_rank_losses(c1, yy).mean() - R.positive_rank_losses(a0, yy).mean())
            )
            per_seed_neg.append(
                float(R.negative_rank_losses(c1, yy).mean() - R.negative_rank_losses(a0, yy).mean())
            )
        d_pos.append(float(np.mean(per_seed_pos)))
        d_neg.append(float(np.mean(per_seed_neg)))
    return {
        "status": "OK",
        "n_boot": n_boot,
        "delta_positive_rank_loss": C.quantile_ci(d_pos),
        "delta_negative_rank_loss": C.quantile_ci(d_neg),
        "delta_roc_auc": C.quantile_ci([-x for x in d_pos]),
        "frac_boot_positive_rank_loss_delta_gt0": float(np.mean(np.asarray(d_pos) > 0)),
    }


def _bootstrap_correlation(
    values: np.ndarray,
    model: str,
    units: dict[tuple[str, int], np.ndarray],
    y: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, Any]:
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    point_rl = np.mean(
        [R.per_sample_rank_loss(units[(model, s)], y) for s in C.MOLECULE_SEEDS], axis=0
    )
    point = {
        "all": _spearman(values, point_rl),
        "positive": _spearman(values[pos], point_rl[pos]),
        "negative": _spearman(values[neg], point_rl[neg]),
    }
    reps = {"all": [], "positive": [], "negative": []}
    for _ in range(n_boot):
        take_pos = rng.choice(pos, size=pos.size, replace=True)
        take_neg = rng.choice(neg, size=neg.size, replace=True)
        idx = np.concatenate([take_pos, take_neg])
        yy = y[idx]
        rl = np.mean(
            [R.per_sample_rank_loss(units[(model, s)][idx], yy) for s in C.MOLECULE_SEEDS], axis=0
        )
        vv = values[idx]
        for key, sel in (
            ("all", np.ones(idx.size, dtype=bool)),
            ("positive", yy == 1),
            ("negative", yy == 0),
        ):
            res = _spearman(vv[sel], rl[sel])
            if res is not None:
                reps[key].append(res)
    return {
        "point": point,
        "ci": {k: C.quantile_ci(v) for k, v in reps.items()},
        "n_boot": n_boot,
    }


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    val = spearmanr(x, y).statistic
    return None if not np.isfinite(val) else float(val)


def main() -> int:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    features, struct_meta = S.build_all()
    feat = pd.DataFrame(features)
    units, rows, y = _load_units()
    if not np.array_equal(rows, np.asarray([r["ogb_row"] for r in features], dtype=np.int64)):
        raise RuntimeError("MolHIV prediction rows and OGB test rows differ")
    if y.size != len(features):
        raise RuntimeError("feature/prediction length mismatch")

    masks = _group_masks(feat)

    # per-unit metrics and rank-identity checks
    per_unit: dict[str, Any] = {}
    identity_max = 0.0
    for model in C.MODELS:
        for seed in C.MOLECULE_SEEDS:
            summ = _unit_summaries(units[(model, seed)], y, masks)
            per_unit[f"{model}:s{seed}"] = summ
            identity_max = max(identity_max, float(summ["overall"]["rank_identity_abs_error"]))
    if identity_max > 1e-12:
        raise RuntimeError(f"rank identity failed: {identity_max}")

    # per-sample table (gitignored intermediate)
    per_sample = feat.copy()
    per_sample["y"] = y.astype(int)
    for model in C.MODELS:
        for seed in C.MOLECULE_SEEDS:
            per_sample[f"score_{model}_s{seed}"] = units[(model, seed)]
            per_sample[f"rank_loss_{model}_s{seed}"] = R.per_sample_rank_loss(units[(model, seed)], y)
    C.LOCAL.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(C.LOCAL / "molhiv_per_sample.csv", index=False)

    # group support table
    group_rows: list[dict[str, Any]] = []
    for dim, groups in masks.items():
        feature_col = {
            "wl2_unseen": "fraction_wl2_unseen",
            "train_tanimoto": "max_train_tanimoto",
            "heavy_atoms": "n_heavy_atoms",
        }[dim]
        for name, mask in groups.items():
            idx = np.flatnonzero(mask)
            npos = int((y[idx] == 1).sum()) if idx.size else 0
            nneg = int((y[idx] == 0).sum()) if idx.size else 0
            group_rows.append(
                {
                    "group_dim": dim,
                    "group": name,
                    "n": int(idx.size),
                    "n_pos": npos,
                    "n_neg": nneg,
                    "positive_rate": float(npos / idx.size) if idx.size else None,
                    "support_status": "OK" if _support_ok(npos, nneg) else "LOW_SUPPORT",
                    "feature_mean": float(feat[feature_col].iloc[idx].mean()) if idx.size else None,
                    "feature_min": float(feat[feature_col].iloc[idx].min()) if idx.size else None,
                    "feature_max": float(feat[feature_col].iloc[idx].max()) if idx.size else None,
                }
            )
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(C.ANALYSIS / "molhiv_support_groups.csv", index=False)

    # long group metrics per model/seed
    metric_rows: list[dict[str, Any]] = []
    for dim, groups in masks.items():
        for name, mask in groups.items():
            idx = np.flatnonzero(mask)
            npos = int((y[idx] == 1).sum()) if idx.size else 0
            nneg = int((y[idx] == 0).sum()) if idx.size else 0
            ok = _support_ok(npos, nneg) and idx.size > 0 and np.unique(y[idx]).size == 2
            for model in C.MODELS:
                for seed in C.MOLECULE_SEEDS:
                    s = per_unit[f"{model}:s{seed}"][dim][name]
                    metric_rows.append(
                        {
                            "scope": "group",
                            "group_dim": dim,
                            "group": name,
                            "model": model,
                            "seed": seed,
                            "n": int(idx.size),
                            "n_pos": npos,
                            "n_neg": nneg,
                            "positive_rate": float(npos / idx.size) if idx.size else None,
                            "support_status": "OK" if ok else "LOW_SUPPORT",
                            "roc_auc": s["roc_auc"] if ok else None,
                            "pr_auc": s["pr_auc"] if ok else None,
                            "positive_rank_loss_mean": s["positive_rank_loss_mean"] if ok else None,
                            "negative_rank_loss_mean": s["negative_rank_loss_mean"] if ok else None,
                        }
                    )
    # overall rows
    for model in C.MODELS:
        for seed in C.MOLECULE_SEEDS:
            s = per_unit[f"{model}:s{seed}"]["overall"]
            metric_rows.append(
                {
                    "scope": "overall",
                    "group_dim": "overall",
                    "group": "all",
                    "model": model,
                    "seed": seed,
                    "n": s["n"],
                    "n_pos": s["n_pos"],
                    "n_neg": s["n_neg"],
                    "positive_rate": s["positive_rate"],
                    "support_status": "OK",
                    "roc_auc": s["roc_auc"],
                    "pr_auc": s["pr_auc"],
                    "positive_rank_loss_mean": s["positive_rank_loss_mean"],
                    "negative_rank_loss_mean": s["negative_rank_loss_mean"],
                }
            )
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(C.ANALYSIS / "molhiv_rank_error_groups.csv", index=False)

    # per-group C1-A0 deltas, seed direction consistency, bootstrap
    rng_global = np.random.default_rng(C.BOOTSTRAP_SEED)
    group_deltas: list[dict[str, Any]] = []
    for dim, groups in masks.items():
        for name, mask in groups.items():
            idx = np.flatnonzero(mask)
            npos = int((y[idx] == 1).sum()) if idx.size else 0
            nneg = int((y[idx] == 0).sum()) if idx.size else 0
            ok = _support_ok(npos, nneg) and idx.size > 0 and np.unique(y[idx]).size == 2
            rec: dict[str, Any] = {
                "group_dim": dim,
                "group": name,
                "n": int(idx.size),
                "n_pos": npos,
                "n_neg": nneg,
                "positive_rate": float(npos / idx.size) if idx.size else None,
                "support_status": "OK" if ok else "LOW_SUPPORT",
                "per_seed": {},
            }
            for seed in C.MOLECULE_SEEDS:
                a = per_unit[f"A0:s{seed}"][dim][name]
                c = per_unit[f"C1:s{seed}"][dim][name]
                if a is None or c is None or not ok:
                    rec["per_seed"][str(seed)] = None
                    continue
                rec["per_seed"][str(seed)] = {
                    "delta_roc_auc": c["roc_auc"] - a["roc_auc"],
                    "delta_positive_rank_loss": c["positive_rank_loss_mean"] - a["positive_rank_loss_mean"],
                    "delta_negative_rank_loss": c["negative_rank_loss_mean"] - a["negative_rank_loss_mean"],
                    "delta_pr_auc": c["pr_auc"] - a["pr_auc"],
                }
            if ok and all(rec["per_seed"].get(str(s)) is not None for s in C.MOLECULE_SEEDS):
                aucs = [rec["per_seed"][str(s)]["delta_roc_auc"] for s in C.MOLECULE_SEEDS]
                rls = [rec["per_seed"][str(s)]["delta_positive_rank_loss"] for s in C.MOLECULE_SEEDS]
                rec["delta_roc_auc_mean"] = float(np.mean(aucs))
                rec["delta_positive_rank_loss_mean"] = float(np.mean(rls))
                rec["delta_roc_auc_seed_sd"] = float(np.std(aucs, ddof=1))
                rec["direction_consistent_auc"] = bool(len({C.sign(x, 1e-12) for x in aucs}) == 1)
                rec["direction_consistent_rank_loss"] = bool(len({C.sign(x, 1e-12) for x in rls}) == 1)
                rec["n_seeds_C1_worse_auc"] = int(sum(x < 0 for x in aucs))
                rec["seed2_extra_delta_auc"] = float(aucs[2] - np.mean(aucs[:2]))
                rec["seed2_is_worst_seed"] = bool(aucs[2] < min(aucs[0], aucs[1]))
                rec["bootstrap"] = _bootstrap_group(
                    mask, y, units, rng_global, C.BOOTSTRAP_REPLICATES
                )
            else:
                rec["delta_roc_auc_mean"] = None
                rec["delta_positive_rank_loss_mean"] = None
                rec["delta_roc_auc_seed_sd"] = None
                rec["direction_consistent_auc"] = None
                rec["direction_consistent_rank_loss"] = None
                rec["n_seeds_C1_worse_auc"] = None
                rec["seed2_extra_delta_auc"] = None
                rec["seed2_is_worst_seed"] = None
                rec["bootstrap"] = {"status": "LOW_SUPPORT"}
            group_deltas.append(rec)

    # size x support crosstab (fixed extreme support groups)
    crosstab_rows: list[dict[str, Any]] = []
    for size_name, size_mask in masks["heavy_atoms"].items():
        for sup_name in ("0", "(0.50,1]"):
            mask = size_mask & masks["wl2_unseen"][sup_name]
            idx = np.flatnonzero(mask)
            npos = int((y[idx] == 1).sum()) if idx.size else 0
            nneg = int((y[idx] == 0).sum()) if idx.size else 0
            ok = _support_ok(npos, nneg) and idx.size > 0 and np.unique(y[idx]).size == 2
            rec = {
                "size_group": size_name,
                "support_group": sup_name,
                "n": int(idx.size),
                "n_pos": npos,
                "n_neg": nneg,
                "positive_rate": float(npos / idx.size) if idx.size else None,
                "support_status": "OK" if ok else "LOW_SUPPORT",
            }
            for model in C.MODELS:
                if not ok:
                    rec[f"{model}_pos_rl_mean"] = None
                    rec[f"{model}_neg_rl_mean"] = None
                    rec[f"{model}_auc_mean"] = None
                else:
                    rls = [
                        per_unit[f"{model}:s{s}"]["heavy_atoms"][size_name]
                        for s in C.MOLECULE_SEEDS
                    ]
                    # recompute on the intersection, not the size-only group
                    pos_means = []
                    neg_means = []
                    aucs = []
                    for seed in C.MOLECULE_SEEDS:
                        summ = R.rank_summary(units[(model, seed)][idx], y[idx])
                        pos_means.append(summ["positive_rank_loss_mean"])
                        neg_means.append(summ["negative_rank_loss_mean"])
                        aucs.append(summ["roc_auc"])
                    rec[f"{model}_pos_rl_mean"] = float(np.mean(pos_means))
                    rec[f"{model}_neg_rl_mean"] = float(np.mean(neg_means))
                    rec[f"{model}_auc_mean"] = float(np.mean(aucs))
            crosstab_rows.append(rec)
    pd.DataFrame(crosstab_rows).to_csv(C.ANALYSIS / "molhiv_size_support_crosstab.csv", index=False)

    # descriptive collapse of the fixed support groups: 0 vs >0, within size strata
    binary_rows: list[dict[str, Any]] = []
    for size_name, size_mask in masks["heavy_atoms"].items():
        for sup_name, sup_mask in (("0", masks["wl2_unseen"]["0"]), (">0", ~masks["wl2_unseen"]["0"])):
            mask = size_mask & sup_mask
            idx = np.flatnonzero(mask)
            npos = int((y[idx] == 1).sum()) if idx.size else 0
            nneg = int((y[idx] == 0).sum()) if idx.size else 0
            ok = _support_ok(npos, nneg) and idx.size > 0 and np.unique(y[idx]).size == 2
            rec = {
                "size_group": size_name,
                "support_group": sup_name,
                "n": int(idx.size),
                "n_pos": npos,
                "n_neg": nneg,
                "positive_rate": float(npos / idx.size) if idx.size else None,
                "support_status": "OK" if ok else "LOW_SUPPORT",
                "note": "descriptive collapse of the pre-defined four support groups; 0 vs any unseen root",
            }
            for model in C.MODELS:
                if idx.size == 0:
                    rec[f"{model}_auc_mean"] = None
                    rec[f"{model}_pos_rl_mean"] = None
                    continue
                aucs = []
                rls = []
                for seed in C.MOLECULE_SEEDS:
                    summ = R.rank_summary(units[(model, seed)][idx], y[idx])
                    aucs.append(summ["roc_auc"])
                    rls.append(summ["positive_rank_loss_mean"])
                rec[f"{model}_auc_mean"] = float(np.mean(aucs)) if ok else None
                rec[f"{model}_pos_rl_mean"] = float(np.mean(rls))
                rec[f"{model}_auc_by_seed"] = [float(x) for x in aucs]
            if ok:
                rec["delta_auc_mean_C1_minus_A0"] = float(rec["C1_auc_mean"] - rec["A0_auc_mean"])
                rec["delta_pos_rl_mean_C1_minus_A0"] = float(rec["C1_pos_rl_mean"] - rec["A0_pos_rl_mean"])
            else:
                rec["delta_auc_mean_C1_minus_A0"] = None
                rec["delta_pos_rl_mean_C1_minus_A0"] = None
            binary_rows.append(rec)
    pd.DataFrame(binary_rows).to_csv(C.ANALYSIS / "molhiv_size_support_binary_crosstab.csv", index=False)

    # correlations with per-molecule rank loss
    corr: dict[str, Any] = {}
    for model in C.MODELS:
        corr[model] = {}
        for feature in ("fraction_wl2_unseen", "fraction_wl2_low_support", "max_train_tanimoto", "n_heavy_atoms", "median_wl2_train_support"):
            corr[model][feature] = _bootstrap_correlation(
                feat[feature].to_numpy(dtype=np.float64),
                model,
                units,
                y,
                np.random.default_rng(C.BOOTSTRAP_SEED),
                C.BOOTSTRAP_REPLICATES,
            )

    # overall seed comparison rows for the deliverable CSV
    seed_rows: list[dict[str, Any]] = []
    for model in C.MODELS:
        for seed in C.MOLECULE_SEEDS:
            s = per_unit[f"{model}:s{seed}"]["overall"]
            seed_rows.append(
                {
                    "scope": "overall",
                    "group_dim": "overall",
                    "group": "all",
                    "model": model,
                    "seed": seed,
                    "n": s["n"],
                    "n_pos": s["n_pos"],
                    "n_neg": s["n_neg"],
                    "roc_auc": s["roc_auc"],
                    "pr_auc": s["pr_auc"],
                    "positive_rank_loss_mean": s["positive_rank_loss_mean"],
                    "negative_rank_loss_mean": s["negative_rank_loss_mean"],
                }
            )
    for rec in group_deltas:
        for seed in C.MOLECULE_SEEDS:
            entry = rec["per_seed"].get(str(seed))
            if entry is None:
                continue
            seed_rows.append(
                {
                    "scope": "group_delta",
                    "group_dim": rec["group_dim"],
                    "group": rec["group"],
                    "model": "C1_minus_A0",
                    "seed": seed,
                    "n": rec["n"],
                    "n_pos": rec["n_pos"],
                    "n_neg": rec["n_neg"],
                    "roc_auc": entry["delta_roc_auc"],
                    "pr_auc": entry["delta_pr_auc"],
                    "positive_rank_loss_mean": entry["delta_positive_rank_loss"],
                    "negative_rank_loss_mean": entry["delta_negative_rank_loss"],
                }
            )
    pd.DataFrame(seed_rows).to_csv(C.ANALYSIS / "molhiv_seed_comparison.csv", index=False)

    # overall paired bootstrap (same stratified scheme)
    overall_boot = _bootstrap_group(np.ones(y.size, dtype=bool), y, units, np.random.default_rng(C.BOOTSTRAP_SEED), C.BOOTSTRAP_REPLICATES)

    payload = {
        "schema": "lcrc-posthoc-molhiv-analysis-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "structure_meta": struct_meta,
        "overall": {k: v["overall"] for k, v in per_unit.items()},
        "overall_pair_bootstrap": overall_boot,
        "group_support": group_rows,
        "group_deltas": group_deltas,
        "size_support_crosstab": crosstab_rows,
        "size_support_binary_crosstab": binary_rows,
        "correlations": corr,
        "correlation_note": (
            "Per-molecule rank loss is computed against ALL opposite-class Test molecules, "
            "not within a structural group.  Its overall Spearman therefore mixes the "
            "positive and negative classes and molecular size; the sign can differ from the "
            "within-group AUC ordering.  Use the group AUC/rank-loss table for within-group "
            "ranking and the within-class correlations for class-specific association."
        ),
        "rank_identity_max_abs_error": identity_max,
        "limitations": [
            "descriptive association on an exposed Test split; not a causal effect of S2",
            "rooted WL2 support is a label-free proxy, not an exact model-representation equivalence class",
            "MolHIV Test has only 130 positives; many fixed groups are LOW_SUPPORT",
            "only three training seeds; row bootstrap does not cover seed variability",
            "official scaffold split means no seen-scaffold comparison group if intersection is empty",
        ],
    }
    C.write_json(C.ANALYSIS / "molhiv_analysis.json", payload)
    print(json.dumps({k: payload["structure_meta"][k] for k in ("n_test_scaffolds_seen_in_train", "n_test_scaffolds", "seen_scaffold_group_available", "n_morgan_failures")}, indent=1))
    print("overall:")
    for k, v in payload["overall"].items():
        print(k, f"AUC={v['roc_auc']:.6f}", f"PR={v['pr_auc']:.6f}", f"posRL={v['positive_rank_loss_mean']:.6f}", f"negRL={v['negative_rank_loss_mean']:.6f}")
    print("group support:")
    for g in group_rows:
        print(g["group_dim"], g["group"], g["n"], g["n_pos"], g["n_neg"], g["support_status"])
    print("group deltas (OK only):")
    for g in group_deltas:
        if g["support_status"] == "OK":
            b = g["bootstrap"]
            print(g["group_dim"], g["group"], f"dAUC={g['delta_roc_auc_mean']:+.4f}", f"CI=[{b['delta_roc_auc']['lo']:+.4f},{b['delta_roc_auc']['hi']:+.4f}]", "consistent", g["direction_consistent_auc"], "seedSD", round(g["delta_roc_auc_seed_sd"],4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
