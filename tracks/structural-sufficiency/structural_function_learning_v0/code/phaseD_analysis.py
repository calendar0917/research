"""Phase D — unified comparison, paired bootstrap contrasts and diagnostics.

Reads only stored predictions (new models in this round's ``analysis/predictions``
and the frozen historical prediction tables).  Recomputes every metric from those
arrays, runs 1000 paired bootstraps (scaffold for MolHIV, molecule for ZINC), copies
the curves, and writes ``metrics.csv`` / ``contrasts.json`` / ``diagnostics.json``.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

import sfl_common as K

N_BOOT = 1000


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------

HIV_MODELS = {
    "COUNT-ALL": {"readout": "linear", "interactions": "none", "supervision": "balanced",
                  "role": "historical_reference"},
    "R128": {"readout": "linear", "interactions": "none", "supervision": "balanced",
             "role": "historical_reference"},
    "XGB-ALL": {"readout": "tree", "interactions": "full", "supervision": "natural",
                "role": "historical_reference"},
    "L-BAL-STD": {"readout": "linear", "interactions": "none", "supervision": "balanced",
                  "role": "new"},
    "L-NAT-STD": {"readout": "linear", "interactions": "none", "supervision": "natural",
                  "role": "new"},
    "L-NAT-RAW": {"readout": "linear", "interactions": "none", "supervision": "natural",
                  "role": "new"},
    "H-ADD-TREE": {"readout": "additive tree", "interactions": "none", "supervision": "natural",
                   "role": "new"},
    "H-NN-ADD": {"readout": "additive neural", "interactions": "none", "supervision": "natural",
                 "role": "new"},
    "H-NN-JOINT": {"readout": "MLP", "interactions": "full", "supervision": "natural",
                   "role": "new"},
    "H-NN-JOINT-BAL": {"readout": "MLP", "interactions": "full", "supervision": "balanced",
                       "role": "new"},
}
ZINC_MODELS = {
    "ZX1": {"readout": "tree", "interactions": "full", "role": "historical_reference"},
    "Z-ADD-TREE": {"readout": "additive tree", "interactions": "none", "role": "new"},
    "ZN1": {"readout": "MLP", "interactions": "full", "role": "historical_reference"},
    "Z-NN-ADD": {"readout": "additive neural", "interactions": "none", "role": "new"},
}


def load_hiv_predictions() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for arm in ("COUNT-ALL", "R128", "XGB-ALL"):
        z = np.load(K.ISRD_PRED / f"{arm}.npz")
        out[arm] = z["logit"].astype(np.float64)
    for arm in ("L-BAL-STD", "L-NAT-STD", "L-NAT-RAW", "H-ADD-TREE",
                "H-NN-JOINT", "H-NN-JOINT-BAL", "H-NN-ADD"):
        z = np.load(K.PREDICTIONS / f"{arm}.npz")
        if "logit" in z.files:
            out[arm] = np.asarray(z["logit"], dtype=np.float64)
        elif "pred" in z.files:
            out[arm] = np.asarray(z["pred"], dtype=np.float64)
        else:                      # neural models store per-split best predictions
            pred = np.empty(z["y"].size, dtype=np.float64)
            for split in K.SPLITS:
                pred[z[f"{split}_rows"]] = z[f"pred_{split}"]
            out[arm] = pred
    return out


def load_zinc_predictions() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for arm, path in (("ZX1", K.SGA_RUNS / "phaseC" / "ZX1_pred.npz"),
                      ("ZN1", K.SGA_RUNS / "phaseC" / "ZN1_pred.npz")):
        z = np.load(path)
        pred = np.empty(z["y"].size, dtype=np.float64)
        for split in K.SPLITS:
            pred[z[f"{split}_rows"]] = z[split]
        out[arm] = pred
    for arm in ("Z-ADD-TREE", "Z-NN-ADD"):
        z = np.load(K.PREDICTIONS / f"{arm}.npz")
        if "pred" in z.files:
            out[arm] = np.asarray(z["pred"], dtype=np.float64)
        else:
            pred = np.empty(z["y"].size)
            for split in K.SPLITS:
                pred[z[f"{split}_rows"]] = z[f"pred_{split}"]
            out[arm] = pred
    return out


# ---------------------------------------------------------------------------
# contrasts
# ---------------------------------------------------------------------------

HIV_CONTRASTS = [
    ("L-BAL-STD_C0.0003-vs-COUNT-ALL", "L-BAL-STD_C0.0003", "COUNT-ALL",
     "stronger L2 (C=0.0003) vs historical C=0.03"),
    ("L-BAL-STD_C0.003-vs-COUNT-ALL", "L-BAL-STD_C0.003", "COUNT-ALL",
     "stronger L2 (C=0.003) vs historical C=0.03"),
    ("L-BAL-STD_selected-vs-COUNT-ALL", "L-BAL-STD", "COUNT-ALL",
     "selected L-BAL-STD vs historical COUNT-ALL"),
    ("L-NAT-STD-vs-L-NAT-RAW", "L-NAT-STD", "L-NAT-RAW",
     "standardised vs raw structure counts, both natural supervision"),
    ("XGB-ALL-vs-H-ADD-TREE", "XGB-ALL", "H-ADD-TREE",
     "full-interaction tree minus additive tree"),
    ("H-NN-JOINT-vs-H-NN-ADD", "H-NN-JOINT", "H-NN-ADD",
     "joint MLP minus additive neural"),
    ("H-ADD-TREE-vs-H-NN-ADD", "H-ADD-TREE", "H-NN-ADD",
     "additive tree minus additive neural"),
    ("XGB-ALL-vs-H-NN-JOINT", "XGB-ALL", "H-NN-JOINT",
     "full tree minus joint MLP"),
    ("H-NN-JOINT-vs-H-NN-JOINT-BAL", "H-NN-JOINT", "H-NN-JOINT-BAL",
     "natural vs balanced BCE, identical MLP otherwise"),
    ("H-ADD-TREE-vs-COUNT-ALL", "H-ADD-TREE", "COUNT-ALL",
     "additive tree minus full linear"),
    ("H-NN-ADD-vs-COUNT-ALL", "H-NN-ADD", "COUNT-ALL",
     "additive neural minus full linear"),
    ("XGB-ALL-vs-COUNT-ALL", "XGB-ALL", "COUNT-ALL",
     "historical full tree minus full linear"),
    ("XGB-ALL-vs-H-NN-ADD", "XGB-ALL", "H-NN-ADD",
     "full tree minus additive neural"),
]
ZINC_CONTRASTS = [
    ("Z-ADD-TREE-vs-ZX1", "Z-ADD-TREE", "ZX1",
     "additive tree vs full-interaction tree; positive gain = additive lower MAE"),
    ("Z-NN-ADD-vs-ZN1", "Z-NN-ADD", "ZN1",
     "additive neural vs joint MLP; positive gain = additive lower MAE"),
    ("Z-NN-ADD-vs-Z-ADD-TREE", "Z-NN-ADD", "Z-ADD-TREE",
     "additive neural vs additive tree; positive gain = neural lower MAE"),
]


def hiv_boot(names: dict[str, np.ndarray], data: K.HIVData, a: str, b: str,
             split: str) -> dict[str, Any]:
    code = data.scaffold_code
    idx = data.rows[split]
    res = K.scaffold_bootstrap(idx, data.y, names[a], names[b], code,
                               n_boot=N_BOOT, seed=K.SEED)
    res.update({"delta_direction": "AUC(A)-AUC(B); AP(A)-AP(B)",
                "a": a, "b": b, "split": split,
                "evidence_label": "DEVELOPMENT_TEST" if split == "test" else "VALID"})
    return res


def zinc_boot(names: dict[str, np.ndarray], data: K.ZincData, a: str, b: str,
              split: str) -> dict[str, Any]:
    idx = data.rows[split]
    res = K.molecule_bootstrap_mae(data.y, names[a], names[b], idx,
                                   n_boot=N_BOOT, seed=K.SEED)
    res.update({"delta_direction": "MAE(B)-MAE(A); positive = A lower MAE",
                "a": a, "b": b, "split": split,
                "evidence_label": "DEVELOPMENT_TEST" if split == "test" else "VALID"})
    return res


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------


def diagnostics(hiv_pred: dict[str, np.ndarray], zinc_pred: dict[str, np.ndarray],
                data_h: K.HIVData, data_z: K.ZincData) -> dict[str, Any]:
    phase_c = K.read_json(K.ISRD_PHASEC)
    full_train_auc = float(phase_c["xgb_all"]["metrics"]["train"]["roc_auc"])
    full_valid_auc = float(phase_c["xgb_all"]["metrics"]["valid"]["roc_auc"])
    zx1 = K.read_json(K.SGA / "runs" / "phaseC" / "ZX1.json")["metrics"]
    zx1_train, zx1_valid = float(zx1["train"]["mae"]), float(zx1["valid"]["mae"])
    out: dict[str, Any] = {
        "hiv": {"full_tree_train_auc": full_train_auc,
                "full_tree_valid_auc": full_valid_auc,
                "full_tree_train_valid_gap": full_train_auc - full_valid_auc},
        "zinc": {"full_tree_train_mae": zx1_train, "full_tree_valid_mae": zx1_valid,
                 "full_tree_valid_train_gap": zx1_valid - zx1_train},
        "models": {},
    }
    b = K.read_json(K.RUNS / "H-ADD-TREE.json")
    add = b["metrics_by_rounds"]["500"]
    add_train = float(add["train"]["roc_auc"])
    cap = (full_train_auc - add_train) > (full_train_auc - full_valid_auc)
    out["models"]["H-ADD-TREE"] = {
        "train_roc_auc": add_train, "valid_roc_auc": float(add["valid"]["roc_auc"]),
        "test_roc_auc": float(add["test"]["roc_auc"]),
        "train_valid_gap": add_train - float(add["valid"]["roc_auc"]),
        "train_fit_ratio_vs_full_tree": add_train / full_train_auc,
        "capacity_confounded": bool(cap),
        "capacity_rule": ("(full_train - add_train) > (full_train - full_valid); "
                          "pre-registered in PROTOCOL §5"),
        "rounds": {k: {s: v[s] for s in K.SPLITS}
                   for k, v in b["metrics_by_rounds"].items()},
        "n_nonempty_trees": b["tree_stats"]["n_nonempty_trees"],
        "n_leaves_total": b["tree_stats"]["n_leaves_total"],
        "n_distinct_features_used": b["tree_stats"]["n_distinct_features_used"],
    }
    bz = K.read_json(K.RUNS / "Z-ADD-TREE.json")
    addz = bz["metrics_by_rounds"]["500"]
    addz_train = float(addz["train"]["mae"])
    capz = (addz_train - zx1_train) > (zx1_valid - zx1_train)
    out["models"]["Z-ADD-TREE"] = {
        "train_mae": addz_train, "valid_mae": float(addz["valid"]["mae"]),
        "test_mae": float(addz["test"]["mae"]),
        "valid_train_gap": float(addz["valid"]["mae"]) - addz_train,
        "train_fit_ratio_vs_full_tree": addz_train / zx1_train,
        "capacity_confounded": bool(capz),
        "capacity_rule": ("(add_train_mae - full_train_mae) > (full_valid_mae - full_train_mae); "
                          "pre-registered in PROTOCOL §5"),
        "rounds": {k: {s: v[s] for s in K.SPLITS}
                   for k, v in bz["metrics_by_rounds"].items()},
        "n_nonempty_trees": bz["tree_stats"]["n_nonempty_trees"],
        "n_leaves_total": bz["tree_stats"]["n_leaves_total"],
        "n_distinct_features_used": bz["tree_stats"]["n_distinct_features_used"],
    }
    for arm in ("H-NN-JOINT", "H-NN-JOINT-BAL", "H-NN-ADD", "Z-NN-ADD"):
        p = K.read_json(K.ANALYSIS / f"phaseC_{arm}.json")
        curve = p["curve"]
        best = int(p["best_epoch"])
        last = curve[-1]
        first = curve[0]
        key = "roc_auc" if arm.startswith("H-") else "mae"
        out["models"][arm] = {
            "best_epoch": best, "last_epoch": int(last["epoch"]),
            "params_total": p["parameters"]["total"],
            "best_epoch_metrics": {s: p["metrics"][s] for s in K.SPLITS},
            "first_epoch_valid": float(first[f"valid_{key}"]),
            "best_epoch_valid": float(curve[best - 1][f"valid_{key}"]),
            "last_epoch_valid": float(last[f"valid_{key}"]),
            "best_minus_last_valid": float(curve[best - 1][f"valid_{key}"]
                                           - last[f"valid_{key}"]),
            "last_epoch_train": float(last[f"train_{key}"]),
            "last_epoch_test": float(last[f"test_{key}"]),
            "train_valid_gap_at_best": float(p["metrics"]["train"][key]
                                             - p["metrics"]["valid"][key]),
            "additive_checks": p.get("additive_checks"),
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.perf_counter()
    dh = K.HIVData(with_scaffold=True)
    dz = K.ZincData()
    hpred = load_hiv_predictions()
    zpred = load_zinc_predictions()

    # identity checks against the frozen inputs
    yz_hist = np.load(K.SGA_RUNS / "phaseC" / "ZN1_pred.npz")["y"]
    if not np.array_equal(yz_hist, dz.y):
        raise ValueError("ZINC y identity mismatch")
    if not np.array_equal(np.load(K.ISRD_PRED / "COUNT-ALL.npz")["split_code"],
                          dh.split_code()):
        raise ValueError("MolHIV split code identity mismatch")

    # store the reused historical predictions in this round's evidence store
    for arm in ("COUNT-ALL", "R128", "XGB-ALL"):
        np.savez_compressed(K.PREDICTIONS / f"{arm}.npz",
                            logit=hpred[arm], y=dh.y, graph_id=np.arange(dh.n),
                            split_code=dh.split_code(), arm=arm, dataset="hiv",
                            source=K.rel(K.ISRD_PRED / f"{arm}.npz"))
    for arm, path in (("ZX1", K.SGA_RUNS / "phaseC" / "ZX1_pred.npz"),
                      ("ZN1", K.SGA_RUNS / "phaseC" / "ZN1_pred.npz")):
        np.savez_compressed(K.PREDICTIONS / f"{arm}.npz", pred=zpred[arm], y=dz.y,
                            graph_id=np.arange(dz.n),
                            train_rows=dz.rows["train"], valid_rows=dz.rows["valid"],
                            test_rows=dz.rows["test"], arm=arm, dataset="zinc",
                            source=K.rel(path))
    # per-C Phase-A predictions (already written by phaseA_linear) stay as evidence

    # ---------------- metrics table ---------------------------------------
    rows: list[dict[str, Any]] = []
    hmetrics = {a: K.hiv_metrics(hpred[a], dh.y, dh.rows) for a in hpred}
    zmetrics = {a: K.zinc_metrics(zpred[a], dz.y, dz.rows) for a in zpred}
    pa = K.read_json(K.ANALYSIS / "phaseA_linear.json")
    for arm, meta in HIV_MODELS.items():
        for split in K.SPLITS:
            m = hmetrics[arm][split]
            rows.append({"dataset": "MolHIV", "model": arm, "readout": meta["readout"],
                         "interactions": meta["interactions"],
                         "supervision": meta["supervision"], "role": meta["role"],
                         "split": split, "n": m["n"],
                         "roc_auc": m["roc_auc"], "average_precision": m["average_precision"],
                         "mae": "", "rmse": "",
                         "selected_setting": (f"C={pa['results'][arm]['selected_C']:g}"
                                              if arm in pa["results"] else ""),
                         "evidence_label": m["evidence_label"],
                         "source": "recomputed_from_stored_predictions",
                         "note": ""})
    # Phase A all C configurations
    for cfg in pa["all_configs"]:
        arm = cfg["arm"]
        for split in K.SPLITS:
            m = K.hiv_metrics(cfg["logit"] if "logit" in cfg else hpred[arm],
                              dh.y, dh.rows)[split] if "logit" not in cfg else None
            if m is None:
                m = cfg["metrics"][split]
            rows.append({"dataset": "MolHIV", "model": arm,
                         "readout": "linear", "interactions": "none",
                         "supervision": ("balanced" if arm == "L-BAL-STD" else "natural"),
                         "role": "new_configuration", "split": split, "n": m["n"],
                         "roc_auc": m["roc_auc"],
                         "average_precision": m["average_precision"],
                         "mae": "", "rmse": "",
                         "selected_setting": f"C={cfg['C']:g}",
                         "evidence_label": m["evidence_label"],
                         "source": ("historical_reuse" if cfg.get("source") else
                                    "recomputed_from_stored_predictions"),
                         "note": "Phase A C grid"})
    # ADD-TREE rounds
    for arm, ds, file in (("H-ADD-TREE", "MolHIV", "H-ADD-TREE"),
                          ("Z-ADD-TREE", "ZINC", "Z-ADD-TREE")):
        br = K.read_json(K.RUNS / f"{file}.json")
        for r, mm in br["metrics_by_rounds"].items():
            for split in K.SPLITS:
                m = mm[split]
                rows.append({"dataset": ds, "model": arm, "readout": "additive tree",
                             "interactions": "none", "supervision": "natural",
                             "role": "new_configuration", "split": split, "n": m["n"],
                             "roc_auc": m.get("roc_auc", ""),
                             "average_precision": m.get("average_precision", ""),
                             "mae": m.get("mae", ""), "rmse": m.get("rmse", ""),
                             "selected_setting": f"rounds={r}",
                             "evidence_label": ("DEVELOPMENT_TEST" if split == "test"
                                                else ("VALID" if split == "valid" else "TRAIN")),
                             "source": "recomputed_from_stored_predictions",
                             "note": "boosting-round slice"})
    for arm, meta in ZINC_MODELS.items():
        for split in K.SPLITS:
            m = zmetrics[arm][split]
            rows.append({"dataset": "ZINC", "model": arm, "readout": meta["readout"],
                         "interactions": meta["interactions"],
                         "supervision": ("L1 on standardised target"
                                         if arm in ("ZN1", "Z-NN-ADD") else "L1"),
                         "role": meta["role"], "split": split, "n": m["n"],
                         "roc_auc": "", "average_precision": "",
                         "mae": m["mae"], "rmse": m["rmse"],
                         "selected_setting": "", "evidence_label": m["evidence_label"],
                         "source": "recomputed_from_stored_predictions", "note": ""})
    import pandas as pd

    K.ANALYSIS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(K.ANALYSIS / "metrics.csv", index=False)

    # ---------------- contrasts -------------------------------------------
    hpred_with_c = dict(hpred)
    for cfg in pa["all_configs"]:
        tag = f"{cfg['arm']}_C{cfg['C']:g}"
        path = K.PREDICTIONS / f"{tag}.npz"
        if path.exists():
            hpred_with_c[tag] = np.load(path)["logit"]
        elif cfg.get("source"):
            # historical C=0.03 fit: the stored COUNT-ALL logit is exactly this fit
            hpred_with_c[tag] = hpred["COUNT-ALL"]
        else:
            raise FileNotFoundError(path)
    contrasts: dict[str, Any] = {"n_boot": N_BOOT, "seed": K.SEED,
                                 "hiv_group_unit": "Bemis-Murcko scaffold",
                                 "zinc_group_unit": "molecule",
                                 "hiv": [], "zinc": []}
    for name, a, b, desc in HIV_CONTRASTS:
        entry = {"name": name, "description": desc}
        for split in ("valid", "test"):
            entry[split] = hiv_boot(hpred_with_c, dh, a, b, split)
        contrasts["hiv"].append(entry)
    for name, a, b, desc in ZINC_CONTRASTS:
        entry = {"name": name, "description": desc}
        for split in ("valid", "test"):
            entry[split] = zinc_boot(zpred, dz, a, b, split)
        contrasts["zinc"].append(entry)
    K.write_json(K.ANALYSIS / "contrasts.json", contrasts)

    # ---------------- curves ----------------------------------------------
    K.CURVES.mkdir(parents=True, exist_ok=True)
    for arm in ("H-NN-JOINT", "H-NN-JOINT-BAL", "H-NN-ADD", "Z-NN-ADD"):
        pass  # curves already written by phaseC
    for arm in ("H-ADD-TREE", "Z-ADD-TREE"):
        br = K.read_json(K.RUNS / f"{arm}.json")
        import csv
        with (K.CURVES / f"{arm}_rounds.csv").open("w", newline="") as fh:
            keys = ["rounds", "split", "roc_auc", "average_precision", "mae", "rmse", "n"]
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r, mm in br["metrics_by_rounds"].items():
                for split in K.SPLITS:
                    m = mm[split]
                    w.writerow({"rounds": r, "split": split,
                                "roc_auc": m.get("roc_auc", ""),
                                "average_precision": m.get("average_precision", ""),
                                "mae": m.get("mae", ""), "rmse": m.get("rmse", ""),
                                "n": m["n"]})
    refs = {
        "ref_H-N-CF_curve.csv": K.SGA / "analysis" / "curves" / "phaseB_N-CF_curve.csv",
        "ref_H-N0_curve.csv": K.SGA / "analysis" / "curves" / "phaseB_N0_curve.csv",
        "ref_Z-N1_curve.csv": K.SGA / "analysis" / "curves" / "phaseC_ZN1_curve.csv",
        "ref_Z-N0_curve.csv": K.SGA / "analysis" / "curves" / "phaseC_ZN0_curve.csv",
    }
    for dst, src in refs.items():
        if src.exists():
            shutil.copyfile(src, K.CURVES / dst)

    # ---------------- diagnostics -----------------------------------------
    diag = diagnostics(hpred, zpred, dh, dz)
    K.write_json(K.ANALYSIS / "diagnostics.json", diag)

    summary = {
        "matched_closed_loop": True,
        "hiv_metrics": hmetrics,
        "zinc_metrics": zmetrics,
        "n_contrasts_hiv": len(contrasts["hiv"]),
        "n_contrasts_zinc": len(contrasts["zinc"]),
    }
    K.write_json(K.ANALYSIS / "phaseD_summary.json", summary)
    print(f"[D] metrics rows={len(rows)} contrasts={len(contrasts['hiv'])}/"
          f"{len(contrasts['zinc'])} in {time.perf_counter() - t0:.1f}s")
    for arm in ("L-BAL-STD", "L-NAT-STD", "L-NAT-RAW", "H-ADD-TREE",
                "H-NN-ADD", "H-NN-JOINT", "H-NN-JOINT-BAL"):
        m = hmetrics[arm]
        print(f"[D] {arm}: valid AUC={m['valid']['roc_auc']:.4f} AP={m['valid']['average_precision']:.4f} "
              f"test AUC={m['test']['roc_auc']:.4f} AP={m['test']['average_precision']:.4f} "
              f"train AUC={m['train']['roc_auc']:.4f}")
    for arm in ("Z-ADD-TREE", "Z-NN-ADD"):
        m = zmetrics[arm]
        print(f"[D] {arm}: valid MAE={m['valid']['mae']:.4f} test MAE={m['test']['mae']:.4f} "
              f"train MAE={m['train']['mae']:.4f}")


if __name__ == "__main__":
    main()
