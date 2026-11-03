"""P1-C: low-cost ESOL size/error diagnostics and NCI1 size-subgroup diagnostics."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import StratifiedKFold, train_test_split

import ph_common as C
import ph_structures as S

ESOL_CSV = (
    C.REPO
    / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench/ESOL/ESOL.csv.gz"
)
ESOL_SPLIT = (
    C.REPO
    / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v1/analysis/source/chembench/ESOL/ESOL.random122.ind.pkl"
)
SIZE_BINS = (("<=15", 15), ("16-25", 25), (">25", None))
SIM_BINS = (("[0,0.30)", 0.30), ("[0.30,0.50)", 0.50), ("[0.50,0.70)", 0.70), ("[0.70,1]", None))
NCI1_RAW = C.REPO / "data/TUD/NCI1/NCI1/raw"


def _size_group(n: int) -> str:
    if n <= 15:
        return "<=15"
    if n <= 25:
        return "16-25"
    return ">25"


def _sim_group(x: float) -> str:
    if x < 0.30:
        return "[0,0.30)"
    if x < 0.50:
        return "[0.30,0.50)"
    if x < 0.70:
        return "[0.50,0.70)"
    return "[0.70,1]"


def _esol_features() -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(ESOL_CSV, usecols=["smiles"])
    smiles = [str(x) for x in frame["smiles"].tolist()]
    split = pd.read_pickle(ESOL_SPLIT)
    train_rows = [int(x) for x in list(split.iloc[0])]
    valid_rows = [int(x) for x in list(split.iloc[1])]
    test_rows = [int(x) for x in list(split.iloc[2])]
    z = C.load_mol("ESOL", "A0", 0)
    pred_rows = np.asarray(z["rows"], dtype=np.int64).tolist()
    if pred_rows != test_rows:
        raise RuntimeError("ESOL prediction rows differ from frozen random122 test rows")
    heavy = []
    parse_modes = []
    for i in test_rows:
        mol = Chem.MolFromSmiles(smiles[i])
        mode = "sanitized"
        if mol is None:
            mol, mode = S.parse_smiles(smiles[i])
        parse_modes.append(mode)
        heavy.append(int(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() != 1)))
    feat = pd.DataFrame(
        {
            "test_index": test_rows,
            "n_heavy_atoms": heavy,
            "size_group": [_size_group(n) for n in heavy],
            "parse_mode": parse_modes,
        }
    )
    # Optional, lower-priority similarity diagnostic using the same label-free tool.
    train_smi = [smiles[i] for i in train_rows]
    test_smi = [smiles[i] for i in test_rows]
    support, support_meta = S.train_support(train_smi)
    rows, struct_meta = S.test_features(test_smi, train_smi, support)
    feat["max_train_tanimoto"] = [r["max_train_tanimoto"] for r in rows]
    feat["sim_group"] = [_sim_group(x) for x in feat["max_train_tanimoto"]]
    meta = {
        "source_csv": C.file_record(ESOL_CSV, "frozen ChemBench ESOL smiles column"),
        "split_file": C.file_record(ESOL_SPLIT, "frozen random122 split indices"),
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "n_test": len(test_rows),
        "test_rows_sha256": C.index_sha256(test_rows),
        "parse_modes": {m: parse_modes.count(m) for m in sorted(set(parse_modes))},
        "support_meta": support_meta,
        "struct_meta": {k: struct_meta[k] for k in ("n_morgan_failures", "rdkit_version", "smiles_sha256")},
    }
    return feat, meta


def _esol() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    feat, meta = _esol_features()
    units = {}
    y = None
    for seed in C.MOLECULE_SEEDS:
        for model in C.MODELS:
            z = C.load_mol("ESOL", model, seed)
            units[(model, seed)] = np.asarray(z["pred"], dtype=np.float64)
            yy = np.asarray(z["y_true"], dtype=np.float64).reshape(-1)
            y = yy if y is None else y
    rows: list[dict[str, Any]] = []
    ordered = {
        "heavy_atoms": ["<=15", "16-25", ">25"],
        "train_tanimoto": ["[0,0.30)", "[0.30,0.50)", "[0.50,0.70)", "[0.70,1]"],
    }
    for dim, col in (("heavy_atoms", "size_group"), ("train_tanimoto", "sim_group")):
        for group in ordered[dim]:
            idx = np.flatnonzero((feat[col] == group).to_numpy())
            rec: dict[str, Any] = {
                "group_dim": dim,
                "group": group,
                "n": int(idx.size),
                "support_status": "OK" if int(idx.size) >= 10 else "LOW_SUPPORT",
                "feature_mean": float(feat.loc[idx, "n_heavy_atoms" if dim == "heavy_atoms" else "max_train_tanimoto"].mean()) if idx.size else None,
            }
            for model in C.MODELS:
                rmses = []
                maes = []
                for seed in C.MOLECULE_SEEDS:
                    p = C.denormalise(units[(model, seed)][idx], "ESOL").reshape(-1)
                    yy = y[idx]
                    rmses.append(float(np.sqrt(np.mean((p - yy) ** 2))))
                    maes.append(float(np.mean(np.abs(p - yy))))
                rec[f"{model}_RMSE"] = float(np.mean(rmses))
                rec[f"{model}_MAE"] = float(np.mean(maes))
                rec[f"{model}_RMSE_sd_over_seeds"] = float(np.std(rmses, ddof=1))
                rec[f"{model}_MAE_sd_over_seeds"] = float(np.std(maes, ddof=1))
            for metric in ("RMSE", "MAE"):
                deltas = []
                for seed in C.MOLECULE_SEEDS:
                    p0 = C.denormalise(units[("A0", seed)][idx], "ESOL").reshape(-1)
                    p1 = C.denormalise(units[("C1", seed)][idx], "ESOL").reshape(-1)
                    yy = y[idx]
                    if metric == "RMSE":
                        f = lambda p: float(np.sqrt(np.mean((p - yy) ** 2)))
                    else:
                        f = lambda p: float(np.mean(np.abs(p - yy)))
                    deltas.append(f(p1) - f(p0))
                rec[f"delta_{metric}_s0"] = deltas[0]
                rec[f"delta_{metric}_s1"] = deltas[1]
                rec[f"delta_{metric}_s2"] = deltas[2]
                rec[f"delta_{metric}_mean"] = float(np.mean(deltas))
                rec[f"delta_{metric}_sd_over_seeds"] = float(np.std(deltas, ddof=1))
            rows.append(rec)

    # Error concentration and overlap of high-error sets.
    concentration = []
    ae = {}
    for model in C.MODELS:
        ae[model] = []
        for seed in C.MOLECULE_SEEDS:
            p = C.denormalise(units[(model, seed)], "ESOL").reshape(-1)
            err = np.abs(p - y)
            k = max(1, int(np.ceil(0.10 * err.size)))
            top = np.argsort(-err)[:k]
            sq_share = float((err[top] ** 2).sum() / (err**2).sum())
            abs_share = float(err[top].sum() / err.sum())
            concentration.append(
                {
                    "model": model,
                    "seed": seed,
                    "n": int(err.size),
                    "top_fraction": 0.10,
                    "top_n": int(k),
                    "top_abs_error_threshold": float(err[top].min()),
                    "share_of_squared_error": sq_share,
                    "share_of_absolute_error": abs_share,
                    "RMSE": float(np.sqrt(np.mean(err**2))),
                    "MAE": float(err.mean()),
                }
            )
            ae[model].append(err)
    # fixed high-error set from A0 seed-mean absolute error, then describe C1 on it
    a0_mean = np.mean(np.asarray(ae["A0"]), axis=0)
    c1_mean = np.mean(np.asarray(ae["C1"]), axis=0)
    k = max(1, int(np.ceil(0.10 * a0_mean.size)))
    high = set(np.argsort(-a0_mean)[:k].tolist())
    overlap = len(high & set(np.argsort(-c1_mean)[:k].tolist()))
    high_idx = np.asarray(sorted(high), dtype=np.int64)
    rest_idx = np.asarray([i for i in range(a0_mean.size) if i not in high], dtype=np.int64)
    fixed = {}
    for model in C.MODELS:
        for region, idx in (("high_error_A0_top10", high_idx), ("remaining_90", rest_idx)):
            rmses = []
            maes = []
            for seed in C.MOLECULE_SEEDS:
                p = C.denormalise(units[(model, seed)][idx], "ESOL").reshape(-1)
                yy = y[idx]
                rmses.append(float(np.sqrt(np.mean((p - yy) ** 2))))
                maes.append(float(np.mean(np.abs(p - yy))))
            fixed[f"{model}/{region}"] = {
                "n": int(idx.size),
                "RMSE_mean": float(np.mean(rmses)),
                "MAE_mean": float(np.mean(maes)),
            }
    meta_out = {
        "features": meta,
        "concentration": concentration,
        "high_error_overlap": {
            "definition": "top 10% by A0 seed-mean absolute error; C1 top-10% computed by C1 seed-mean absolute error",
            "k": int(k),
            "overlap_count": int(overlap),
            "jaccard": float(overlap / (2 * k - overlap)) if k else None,
            "A0_top10_indices": sorted(int(x) for x in high),
            "fixed_region_metrics": fixed,
        },
        "note": "post-hoc descriptive; no sample is removed from the main ESOL score",
    }
    return rows, concentration, meta_out


def _nci1_node_counts() -> np.ndarray:
    gi = np.loadtxt(NCI1_RAW / "NCI1_graph_indicator.txt", dtype=np.int64)
    n_graphs = int(gi.max())
    return np.bincount(gi, minlength=n_graphs + 1)[1:]


def _nci1_folds() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = np.loadtxt(NCI1_RAW / "NCI1_graph_labels.txt", dtype=np.int64)
    manifest = C.read_json(C.PTB / "analysis/split_manifest.json")["folds"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261102)
    splits = list(skf.split(np.zeros(y.size, dtype=np.int8), y))
    folds = []
    for fold, (tr, te) in enumerate(splits):
        tr = np.sort(np.asarray(tr, dtype=np.int64))
        te = np.sort(np.asarray(te, dtype=np.int64))
        rec = next(x for x in manifest if x["dataset"] == "NCI1" and int(x["fold"]) == fold)
        y_ot = y[tr]
        inner_seed = 20261102 + fold
        a, b = train_test_split(
            np.arange(tr.size, dtype=np.int64),
            test_size=0.10,
            stratify=y_ot,
            random_state=inner_seed,
        )
        train = np.sort(tr[a])
        valid = np.sort(tr[b])
        checks = {
            "train_sha256": C.index_sha256(train.tolist()) == rec["sha256"]["train"],
            "valid_sha256": C.index_sha256(valid.tolist()) == rec["sha256"]["valid"],
            "test_sha256": C.index_sha256(te.tolist()) == rec["sha256"]["test"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"NCI1 fold {fold} reconstruction hash mismatch: {checks}")
        folds.append(
            {
                "fold": fold,
                "train": train,
                "valid": valid,
                "test": te,
                "y": y,
                "checks": checks,
                "n_train": int(train.size),
                "n_valid": int(valid.size),
                "n_test": int(te.size),
            }
        )
    meta = {
        "n_graphs": int(y.size),
        "n_splits": 5,
        "fold_seed": 20261102,
        "inner_valid_seed_base": 20261102,
        "inner_valid_fraction": 0.10,
        "folds": [
            {
                "fold": f["fold"],
                "n_train": f["n_train"],
                "n_valid": f["n_valid"],
                "n_test": f["n_test"],
                "checks": f["checks"],
            }
            for f in folds
        ],
        "source_files": {
            "graph_indicator": C.file_record(NCI1_RAW / "NCI1_graph_indicator.txt", "TUD node->graph ids; node counts only"),
            "graph_labels": C.file_record(NCI1_RAW / "NCI1_graph_labels.txt", "TUD graph labels; fold reconstruction only"),
        },
    }
    return folds, meta


def _nci1() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = _nci1_node_counts()
    folds, meta = _nci1_folds()
    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    fold_thresholds = []
    for f in folds:
        fold = f["fold"]
        train_counts = counts[f["train"]]
        q1, q2 = np.quantile(train_counts, [1 / 3, 2 / 3], method="linear")
        fold_thresholds.append(
            {
                "fold": fold,
                "q33_nodes": float(q1),
                "q67_nodes": float(q2),
                "train_node_min": int(train_counts.min()),
                "train_node_max": int(train_counts.max()),
                "train_node_median": float(np.median(train_counts)),
            }
        )
        test_counts = counts[f["test"]].astype(np.int64)
        groups = np.where(test_counts <= q1, "small", np.where(test_counts <= q2, "medium", "large"))
        for model in C.MODELS:
            for seed in C.TU_SEEDS:
                z = C.load_npz(C.tu_run_dir("NCI1", fold, seed, model) / "test_predictions.npz")
                gids = np.asarray(z["gids"], dtype=np.int64)
                y = np.asarray(z["y_true"], dtype=np.int64)
                pred = np.argmax(np.asarray(z["logits"], dtype=np.float64), axis=1)
                if not np.array_equal(gids, f["test"]):
                    raise RuntimeError(f"NCI1 f{fold}/s{seed}/{model}: gids != frozen test")
                for group in ("small", "medium", "large"):
                    sel = groups == group
                    n = int(sel.sum())
                    correct = int((pred[sel] == y[sel]).sum())
                    rows.append(
                        {
                            "group": group,
                            "fold": fold,
                            "seed": seed,
                            "model": model,
                            "n": n,
                            "n_correct": correct,
                            "accuracy": float(correct / n) if n else None,
                            "mean_nodes": float(test_counts[sel].mean()) if n else None,
                            "node_min": int(test_counts[sel].min()) if n else None,
                            "node_max": int(test_counts[sel].max()) if n else None,
                        }
                    )
        # paired accuracy and error counts
        for seed in C.TU_SEEDS:
            z0 = C.load_npz(C.tu_run_dir("NCI1", fold, seed, "A0") / "test_predictions.npz")
            z1 = C.load_npz(C.tu_run_dir("NCI1", fold, seed, "C1") / "test_predictions.npz")
            y0 = np.asarray(z0["y_true"], dtype=np.int64)
            y1 = np.asarray(z1["y_true"], dtype=np.int64)
            if not np.array_equal(y0, y1):
                raise RuntimeError("NCI1 A0/C1 labels differ")
            p0 = np.argmax(np.asarray(z0["logits"], dtype=np.float64), axis=1) == y0
            p1 = np.argmax(np.asarray(z1["logits"], dtype=np.float64), axis=1) == y0
            for group in ("small", "medium", "large"):
                sel = groups == group
                n = int(sel.sum())
                n11 = int((p0[sel] & p1[sel]).sum())
                n10 = int((p0[sel] & ~p1[sel]).sum())
                n01 = int((~p0[sel] & p1[sel]).sum())
                n00 = int((~p0[sel] & ~p1[sel]).sum())
                paired.append(
                    {
                        "group": group,
                        "fold": fold,
                        "seed": seed,
                        "n": n,
                        "both_correct": n11,
                        "A0_correct_C1_wrong": n10,
                        "A0_wrong_C1_correct": n01,
                        "both_wrong": n00,
                        "delta_accuracy_C1_minus_A0": float((n01 - n10) / n) if n else None,
                    }
                )
    meta["fold_thresholds"] = fold_thresholds
    return rows, {"meta": meta, "paired": paired}


def main() -> int:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    esol_rows, esol_conc, esol_meta = _esol()
    C.save_csv(C.ANALYSIS / "esol_subgroups.csv", esol_rows, columns=list(esol_rows[0].keys()))
    C.save_csv(C.ANALYSIS / "esol_error_concentration.csv", esol_conc, columns=list(esol_conc[0].keys()))
    C.write_json(C.ANALYSIS / "esol_aux.json", {"analysis": "ESOL size and optional similarity diagnostics", "rows": esol_rows, "concentration": esol_conc, **esol_meta})

    nci_rows, nci_payload = _nci1()
    C.save_csv(C.ANALYSIS / "nci1_subgroups.csv", nci_rows, columns=list(nci_rows[0].keys()))
    C.save_csv(C.ANALYSIS / "nci1_paired_errors.csv", nci_payload["paired"], columns=list(nci_payload["paired"][0].keys()))

    # aggregate by group (pooled over fold/seed; descriptive, not independent)
    agg = {}
    for group in ("small", "medium", "large"):
        rs = [r for r in nci_rows if r["group"] == group]
        agg[group] = {
            "n_graph_units": int(sum(r["n"] for r in rs)),
            "n_units": len(rs),
            "A0_accuracy_pooled": float(sum(r["n_correct"] for r in rs if r["model"] == "A0") / max(1, sum(r["n"] for r in rs if r["model"] == "A0"))),
            "C1_accuracy_pooled": float(sum(r["n_correct"] for r in rs if r["model"] == "C1") / max(1, sum(r["n"] for r in rs if r["model"] == "C1"))),
            "A0_accuracy_mean_over_units": float(np.mean([r["accuracy"] for r in rs if r["model"] == "A0"])),
            "C1_accuracy_mean_over_units": float(np.mean([r["accuracy"] for r in rs if r["model"] == "C1"])),
            "delta_pooled": float(
                sum(r["n_correct"] for r in rs if r["model"] == "C1") / max(1, sum(r["n"] for r in rs if r["model"] == "C1"))
                - sum(r["n_correct"] for r in rs if r["model"] == "A0") / max(1, sum(r["n"] for r in rs if r["model"] == "A0"))
            ),
            "mean_nodes": float(np.mean([r["mean_nodes"] for r in rs])),
        }
        prs = [r for r in nci_payload["paired"] if r["group"] == group]
        agg[group]["paired"] = {
            "A0_correct_C1_wrong": int(sum(r["A0_correct_C1_wrong"] for r in prs)),
            "A0_wrong_C1_correct": int(sum(r["A0_wrong_C1_correct"] for r in prs)),
            "both_correct": int(sum(r["both_correct"] for r in prs)),
            "both_wrong": int(sum(r["both_wrong"] for r in prs)),
            "n_units_C1_lower": int(sum((r["delta_accuracy_C1_minus_A0"] or 0) < 0 for r in prs)),
            "n_units_C1_higher": int(sum((r["delta_accuracy_C1_minus_A0"] or 0) > 0 for r in prs)),
            "n_units_tied": int(sum((r["delta_accuracy_C1_minus_A0"] or 0) == 0 for r in prs)),
            "n_units": len(prs),
        }
    payload = {
        "schema": "lcrc-posthoc-nci1-aux-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "meta": nci_payload["meta"],
        "aggregate_by_group": agg,
        "paired": nci_payload["paired"],
        "limitations": [
            "subgroups are descriptive, not independent benchmarks",
            "same test graphs appear under two training seeds; do not treat them as independent samples",
            "fold-specific tertiles keep groups comparable within a fold, not across folds",
        ],
    }
    C.write_json(C.ANALYSIS / "nci1_aux.json", payload)
    print("ESOL groups")
    for r in esol_rows:
        print(r["group_dim"], r["group"], r["n"], f"A0rmse={r['A0_RMSE']:.4f}", f"C1rmse={r['C1_RMSE']:.4f}", f"dRMSE={r['delta_RMSE_mean']:+.4f}")
    print("ESOL concentration")
    for r in esol_conc:
        print(r["model"], r["seed"], f"top10 sq share={r['share_of_squared_error']:.3f}", f"abs share={r['share_of_absolute_error']:.3f}")
    print("ESOL overlap", esol_meta["high_error_overlap"]["overlap_count"], "of", esol_meta["high_error_overlap"]["k"])
    print("NCI1")
    for g, v in agg.items():
        print(g, f"n={v['n_graph_units']}", f"A0={v['A0_accuracy_pooled']:.4f}", f"C1={v['C1_accuracy_pooled']:.4f}", f"d={v['delta_pooled']:+.4f}", v["paired"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
