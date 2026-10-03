"""Part B — the single frozen topology correction: robust class-median
shrinkage toward zero (local CPU, read-only).

Rule (pre-registered, not tuned, prior strength 5 fixed, never selected on dev):

    per base model m and its own meta fold (F_A -> B, F_B -> A):
        r = y - p_base_cal
        class key c = exact raw graph-only T25 class (round to 1e-6, previous
                      round's cache / rounding rule)
        within c, take the median residual per canonical molecule group first
        (each distinct meta molecule contributes one value)
        m_c = median over those group values
        n_c = number of distinct canonical meta molecules in class c
        q_c = n_c / (n_c + 5) * m_c          (prior strength 5, fixed)
        unseen class -> q = 0
        candidate Q = (p_A + q_A + p_B + q_B) / 2

No prediction-value nearest neighbour, no new input, no ridge, no per-class
clipping, no dev-based selection.  Official test untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd
from tracks.ksvd.experiments.luyin16 import zinc_topology_crossfit_diagnostic_v1 as d

SRC = Path("tracks/ksvd/results/zinc_topology_crossfit_diagnostic_v1").resolve()
OUT = Path(__file__).resolve().parent
FOLDS = ("A", "B")
PRIOR = 5.0
N_BOOT = 1000
BOOT_SEED = 20261003
ROUND = 6


def mae(p, y):
    return float(np.abs(np.asarray(p) - np.asarray(y)).mean())


def paired_boot_fixed(p_a, p_b, y, n_boot=N_BOOT, seed=BOOT_SEED):
    y = np.asarray(y, np.float64)
    ea = np.abs(np.asarray(p_a, np.float64) - y)
    eb = np.abs(np.asarray(p_b, np.float64) - y)
    diff = ea - eb
    n = len(y)
    rng = np.random.default_rng(int(seed))
    draws = np.array([float(diff[rng.integers(0, n, n)].mean()) for _ in range(int(n_boot))])
    return {"point": float(diff.mean()), "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n_boot": int(n_boot), "seed": int(seed)}


def class_keys(t25: np.ndarray, rows: np.ndarray) -> list[tuple]:
    return [tuple(np.round(r, ROUND).tolist()) for r in np.asarray(t25)[rows]]


def build_classes(t25, meta_idx, r_meta, gid):
    """Return {class_key: {"m_c","n_c","q_c","group_residuals":[...]}}."""
    keys = class_keys(t25, meta_idx)
    groups: dict[tuple, dict[int, list[float]]] = {}
    for pos, k in enumerate(keys):
        groups.setdefault(k, {}).setdefault(int(gid[meta_idx[pos]]), []).append(float(r_meta[pos]))
    out = {}
    for k, by_group in groups.items():
        group_med = np.array([float(np.median(v)) for v in by_group.values()])
        m_c = float(np.median(group_med))
        n_c = int(len(by_group))
        out[k] = {"m_c": m_c, "n_c": n_c, "q_c": float(n_c / (n_c + PRIOR) * m_c),
                  "group_medians": group_med.tolist(),
                  "n_rows": int(sum(len(v) for v in by_group.values()))}
    return out


def apply_classes(t25, dev_idx, classes):
    keys = class_keys(t25, dev_idx)
    q = np.array([classes[k]["q_c"] if k in classes else 0.0 for k in keys], np.float64)
    covered = np.array([k in classes for k in keys], bool)
    return q, covered


def group_table(pred, y, pen):
    out = {}
    for name, sel in (("penalty_0", pen == 0), ("penalty_-1", pen == -1), ("penalty_le_-2", pen <= -2)):
        sel = np.asarray(sel)
        err = np.abs(np.asarray(pred)[sel] - np.asarray(y)[sel])
        out[name] = {"n": int(sel.sum()), "mae": float(err.mean()) if sel.any() else None,
                     "contrib": float(err.sum() / len(y))}
    out["overall"] = {"n": int(len(y)), "mae": mae(pred, y)}
    return out


def main() -> int:
    pen, _ = zjd._load_penalties()
    handoff = np.load(zjd.HANDOFF / "train.npz", allow_pickle=True)
    gid = handoff["canonical_group_id"].astype(np.int64)
    idx = d.load_subfold_index()
    dev_idx = idx["dev_idx"]
    dev_pen = pen[dev_idx]
    t25 = d.raw_topology25()

    bases = {f: json.loads((SRC / f"F_{f}.json").read_text()) for f in FOLDS}
    ro = json.loads((SRC / "readouts.json").read_text())
    y = np.asarray(bases["A"]["dev_targets"], np.float64)

    pbase = {f: np.asarray(bases[f]["dev_predictions_cal"], np.float64) for f in FOLDS}
    pconst = {f: np.asarray(ro[f]["dev_predictions_const"], np.float64) for f in FOLDS}
    base = (pbase["A"] + pbase["B"]) / 2.0
    CONST = (pconst["A"] + pconst["B"]) / 2.0

    per_fold = {}
    Q_parts = {}
    classes_meta = {}
    for f in FOLDS:
        meta_idx = idx["b_idx" if f == "A" else "a_idx"]
        r_meta = np.asarray(bases[f]["meta_targets"], np.float64) - np.asarray(
            bases[f]["meta_predictions_cal"], np.float64)
        classes = build_classes(t25, meta_idx, r_meta, gid)
        q_dev, covered = apply_classes(t25, dev_idx, classes)
        Q_f = pbase[f] + q_dev
        Q_parts[f] = Q_f
        classes_meta[f] = classes
        per_fold[f] = {
            "n_classes": int(len(classes)),
            "n_meta_rows": int(len(meta_idx)),
            "n_dev_covered": int(covered.sum()),
            "dev_coverage": float(covered.mean()),
            "q_dev_abs_mean": float(np.abs(q_dev).mean()),
            "q_dev_quantiles": {str(q): float(np.percentile(q_dev, q)) for q in (0, 25, 50, 75, 95, 100)},
            "base_mae": mae(pbase[f], y),
            "Q_mae": mae(Q_f, y),
            "Q_minus_base_gain": float(mae(pbase[f], y) - mae(Q_f, y)),
            "bootstrap_Q_vs_base": paired_boot_fixed(Q_f, pbase[f], y),
            "n_classes_n_ge5": int(sum(1 for v in classes.values() if v["n_c"] >= 5)),
            "n_classes_n_eq1": int(sum(1 for v in classes.values() if v["n_c"] == 1)),
        }

    Q = (Q_parts["A"] + Q_parts["B"]) / 2.0
    main_table = {"base": mae(base, y), "Q": mae(Q, y), "const": mae(CONST, y)}
    gain_base = float(main_table["base"] - main_table["Q"])
    gain_const = float(main_table["const"] - main_table["Q"])

    g0 = dev_pen == 0
    base_g0 = mae(base[g0], y[g0])
    Q_g0 = mae(Q[g0], y[g0])
    const_g0 = mae(CONST[g0], y[g0])

    gate = {
        "gain_Q_vs_base": gain_base,
        "gain_Q_vs_base_ge_0.003": bool(gain_base >= 0.003),
        "gain_Q_vs_const": gain_const,
        "gain_Q_vs_const_ge_0.003": bool(gain_const >= 0.003),
        "per_fold_Q_vs_own_base": {f: per_fold[f]["Q_minus_base_gain"] for f in FOLDS},
        "per_fold_nonneg": {f: bool(per_fold[f]["Q_minus_base_gain"] >= 0.0) for f in FOLDS},
        "G0_base_mae": base_g0, "G0_Q_mae": Q_g0,
        "G0_worsening": float(Q_g0 - base_g0),
        "G0_worsening_le_0.001": bool((Q_g0 - base_g0) <= 0.001),
        "G0_const_mae": const_g0,
    }
    gate["passed"] = bool(gate["gain_Q_vs_base_ge_0.003"] and gate["gain_Q_vs_const_ge_0.003"]
                          and all(gate["per_fold_nonneg"].values()) and gate["G0_worsening_le_0.001"])

    groups = {"base": group_table(base, y, dev_pen), "Q": group_table(Q, y, dev_pen),
              "const": group_table(CONST, y, dev_pen)}
    boot = {"Q_vs_base": paired_boot_fixed(Q, base, y), "Q_vs_const": paired_boot_fixed(Q, CONST, y),
            "Q_vs_base_G0": paired_boot_fixed(Q[g0], base[g0], y[g0]),
            "Q_vs_const_G0": paired_boot_fixed(Q[g0], CONST[g0], y[g0])}

    # uncovered rows and severe rows
    qA, covA = apply_classes(t25, dev_idx, classes_meta["A"])
    qB, covB = apply_classes(t25, dev_idx, classes_meta["B"])
    uncovered = {
        "n_uncovered_A": int((~covA).sum()), "n_uncovered_B": int((~covB).sum()),
        "rows_uncovered_by_A": [int(j) for j in np.where(~covA)[0].tolist()],
        "rows_uncovered_by_B": [int(j) for j in np.where(~covB)[0].tolist()],
    }
    severe = []
    for j in np.where(dev_pen <= -2)[0].tolist():
        severe.append({"dev_row": int(j), "train_index": int(dev_idx[j]), "penalty": int(dev_pen[j]),
                       "y": float(y[j]), "base": float(base[j]), "Q": float(Q[j]),
                       "q_A": float(qA[j]), "q_B": float(qB[j]),
                       "covered_A": bool(covA[j]), "covered_B": bool(covB[j]),
                       "base_err": float(abs(base[j] - y[j])), "Q_err": float(abs(Q[j] - y[j]))})
    ring_rows = []
    for j in np.where(dev_pen < 0)[0].tolist():
        ring_rows.append({"dev_row": int(j), "train_index": int(dev_idx[j]), "penalty": int(dev_pen[j]),
                          "base_err": float(abs(base[j] - y[j])), "Q_err": float(abs(Q[j] - y[j])),
                          "q_A": float(qA[j]), "q_B": float(qB[j]),
                          "covered_A": bool(covA[j]), "covered_B": bool(covB[j])})

    # per-class support summary (compressed)
    class_support = {}
    for f in FOLDS:
        vals = np.array([v["n_c"] for v in classes_meta[f].values()])
        class_support[f] = {"n_classes": int(len(vals)),
                            "n_c_hist": {str(k): int((vals == k).sum()) for k in sorted(set(vals.tolist()))[:15]},
                            "n_c_max": int(vals.max()) if len(vals) else 0,
                            "n_c_median": float(np.median(vals)) if len(vals) else 0.0}

    payload = {
        "rule": {"name": "class-median shrinkage toward zero", "prior_strength": PRIOR,
                 "class_key": "exact raw graph-only T25 class, round 1e-6",
                 "dedupe": "median residual per canonical molecule group within class",
                 "unseen_class": 0.0, "clipping": False, "dev_tuned": False},
        "main_table": main_table, "gain_Q_vs_base": gain_base, "gain_Q_vs_const": gain_const,
        "per_fold": per_fold, "gate": gate, "groups": groups, "bootstrap": boot,
        "uncovered": uncovered, "severe_rows": severe, "ring_rows": ring_rows,
        "class_support": class_support,
        "n_dev": int(len(y)), "dev_strata": {"0": int((dev_pen == 0).sum()),
                                              "-1": int((dev_pen == -1).sum()),
                                              "le_-2": int((dev_pen <= -2).sum())},
        "official_test_loaded": False,
    }
    (OUT / "part_b_class_median_shrinkage.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({"main_table": main_table, "gain_Q_vs_base": gain_base, "gain_Q_vs_const": gain_const,
                      "gate_passed": gate["passed"], "G0_worsening": gate["G0_worsening"],
                      "per_fold": {f: per_fold[f]["Q_minus_base_gain"] for f in FOLDS},
                      "coverage": {f: per_fold[f]["dev_coverage"] for f in FOLDS},
                      "boot": {k: (round(v["point"], 6), round(v["lo"], 6), round(v["hi"], 6))
                               for k, v in boot.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())