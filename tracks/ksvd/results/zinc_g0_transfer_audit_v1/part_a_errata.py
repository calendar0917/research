"""Part A — corrected paired bootstrap + damage statistics for the ExtraTrees
residual readouts of zinc-topology-crossfit-diagnostic-v1 (local CPU, read-only).

The frozen `analyze.py::paired_boot` drew two *independent* row resamples
(`rng.integers` twice), so the two arms were evaluated on different rows and the
per-row pairing was destroyed.  This script fixes that (one `idx` per bootstrap
iteration, resample the per-row difference `d_i = |p_a - y| - |p_b - y|`),
recomputes the three frozen comparisons with the same protocol (1000x,
seed=20261003, average A/B predictions *before* MAE), and adds the pre-registered
verification checks and the P/TP damage statistics.

All inputs are existing arrays.  No model fit, no backward, no official test.
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
N_BOOT = 1000
BOOT_SEED = 20261003


def mae(p, y):
    return float(np.abs(np.asarray(p) - np.asarray(y)).mean())


# ---------------------------------------------------------------------------
# corrected paired bootstrap
# ---------------------------------------------------------------------------


def paired_boot_fixed(p_a, p_b, y, n_boot=N_BOOT, seed=BOOT_SEED):
    """Correct row-paired bootstrap of MAE(a) - MAE(b).

    One bootstrap row index vector per iteration; the paired per-row difference
    is resampled.  Point estimate is the observed mean difference.
    """
    y = np.asarray(y, np.float64)
    ea = np.abs(np.asarray(p_a, np.float64) - y)
    eb = np.abs(np.asarray(p_b, np.float64) - y)
    diff = ea - eb
    n = len(y)
    rng = np.random.default_rng(int(seed))
    obs = float(diff.mean())
    draws = np.empty(int(n_boot), np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)  # ONE index vector for BOTH arms
        draws[i] = float(diff[idx].mean())
    return {
        "point": obs,
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "resampling": "shared row index per iteration; resamples per-row |err| difference",
    }


def paired_boot_broken(p_a, p_b, y, n_boot=N_BOOT, seed=BOOT_SEED):
    """The frozen (buggy) two-independent-resample version, for the errata."""
    y = np.asarray(y, np.float64)
    ea = np.abs(np.asarray(p_a, np.float64) - y)
    eb = np.abs(np.asarray(p_b, np.float64) - y)
    n = len(y)
    rng = np.random.default_rng(int(seed))
    obs = float(ea.mean() - eb.mean())
    draws = np.array([
        float(ea[rng.integers(0, n, n)].mean() - eb[rng.integers(0, n, n)].mean())
        for _ in range(int(n_boot))
    ])
    return {"point": obs, "lo": float(np.percentile(draws, 2.5)),
            "hi": float(np.percentile(draws, 97.5)), "n_boot": int(n_boot), "seed": int(seed),
            "resampling": "BROKEN: two independent row resamples"}


# ---------------------------------------------------------------------------
# verification checks
# ---------------------------------------------------------------------------


def verification(p_a, p_b, y):
    out = {}
    # 1. identical predictions -> gain 0, CI [0,0]
    ident = paired_boot_fixed(p_a, p_a, y)
    out["identical"] = {"boot": ident, "gain_zero": bool(ident["point"] == 0.0),
                        "ci_zero": bool(ident["lo"] == 0.0 and ident["hi"] == 0.0)}
    # 2. swap the two prediction arrays -> gain flips sign, CI mirrors
    fwd = paired_boot_fixed(p_a, p_b, y)
    rev = paired_boot_fixed(p_b, p_a, y)
    out["swap"] = {
        "forward": fwd, "reverse": rev,
        "point_flips": bool(abs(rev["point"] + fwd["point"]) < 1e-12),
        "lo_mirrors_hi": float(rev["lo"] + fwd["hi"]),
        "hi_mirrors_lo": float(rev["hi"] + fwd["lo"]),
        "ok": bool(abs(rev["point"] + fwd["point"]) < 1e-12
                   and abs(rev["lo"] + fwd["hi"]) < 1e-12
                   and abs(rev["hi"] + fwd["lo"]) < 1e-12),
    }
    # 3. constant shift c: every per-row MAE difference and the paired bootstrap
    #    must lie inside +-|c|.
    c = float(abs(np.mean(p_b - p_a))) if False else None  # unused; c set by caller
    return out


def const_bound_check(base, const, y, c):
    """Per-row MAE differences under a constant shift c must be within ±|c|."""
    ea = np.abs(np.asarray(base, np.float64) - np.asarray(y, np.float64))
    eb = np.abs(np.asarray(const, np.float64) - np.asarray(y, np.float64))
    row_diff = ea - eb
    boot = paired_boot_fixed(base, const, y)
    bound = float(abs(c))
    return {
        "shift_c": float(c),
        "row_min": float(row_diff.min()), "row_max": float(row_diff.max()),
        "row_within_bound": bool(row_diff.min() >= -bound - 1e-12 and row_diff.max() <= bound + 1e-12),
        "boot": boot,
        "boot_within_bound": bool(boot["lo"] >= -bound - 1e-12 and boot["hi"] <= bound + 1e-12),
        "bound": bound,
    }


# ---------------------------------------------------------------------------
# damage statistics for P and TP relative to base
# ---------------------------------------------------------------------------


def damage_stats(p_new, base, y):
    y = np.asarray(y, np.float64)
    e_base = np.abs(np.asarray(base, np.float64) - y)
    e_new = np.abs(np.asarray(p_new, np.float64) - y)
    delta = e_new - e_base  # per-row MAE change; >0 = damaged
    order = np.argsort(-delta)  # worst first
    n = len(delta)
    total_damage = float(delta[delta > 0].sum())
    total_gain = float(-delta[delta < 0].sum())
    top10 = float(delta[order[:10]].sum())
    top50 = float(delta[order[:50]].sum())
    return {
        "n": int(n),
        "mae_base": float(e_base.mean()),
        "mae_new": float(e_new.mean()),
        "mae_gain": float(e_base.mean() - e_new.mean()),
        "n_improved": int((delta < 0).sum()),
        "n_worsened": int((delta > 0).sum()),
        "n_tied": int((delta == 0).sum()),
        "total_positive_damage_sum": total_damage,
        "total_negative_gain_sum": total_gain,
        "net_delta_sum": float(delta.sum()),
        "net_delta_mean": float(delta.mean()),
        "top10_damage_sum": top10,
        "top50_damage_sum": top50,
        "top10_share_of_damage": float(top10 / total_damage) if total_damage > 0 else None,
        "top50_share_of_damage": float(top50 / total_damage) if total_damage > 0 else None,
        "delta_quantiles": {q: float(np.percentile(delta, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)},
        "worst_rows": [
            {"dev_row": int(j), "delta_mae": float(delta[j]),
             "base_err": float(e_base[j]), "new_err": float(e_new[j])}
            for j in order[:10].tolist()
        ],
    }


def amplitude_quantiles(p_new, base):
    corr = np.asarray(p_new, np.float64) - np.asarray(base, np.float64)
    a = np.abs(corr)
    return {
        "signed_quantiles": {q: float(np.percentile(corr, q)) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)},
        "abs_quantiles": {q: float(np.percentile(a, q)) for q in (50, 75, 90, 95, 99, 100)},
        "abs_mean": float(a.mean()), "abs_max": float(a.max()),
    }


def main() -> int:
    pen, _ = zjd._load_penalties()
    idx = d.load_subfold_index()
    dev_idx = idx["dev_idx"]
    dev_pen = pen[dev_idx]
    bases = {f: json.loads((SRC / f"F_{f}.json").read_text()) for f in FOLDS}
    ro = json.loads((SRC / "readouts.json").read_text())

    y = np.asarray(bases["A"]["dev_targets"], np.float64)
    assert np.array_equal(y, np.asarray(bases["B"]["dev_targets"]))
    assert np.array_equal(y, np.asarray(ro["A"]["dev_targets"]))

    pbase = {f: np.asarray(bases[f]["dev_predictions_cal"], np.float64) for f in FOLDS}
    pP = {f: np.asarray(ro[f]["dev_predictions_P"], np.float64) for f in FOLDS}
    pTP = {f: np.asarray(ro[f]["dev_predictions_TP"], np.float64) for f in FOLDS}
    pconst = {f: np.asarray(ro[f]["dev_predictions_const"], np.float64) for f in FOLDS}

    base = (pbase["A"] + pbase["B"]) / 2.0
    P = (pP["A"] + pP["B"]) / 2.0
    TP = (pTP["A"] + pTP["B"]) / 2.0
    CONST = (pconst["A"] + pconst["B"]) / 2.0
    const_shift = float((ro["A"]["r_meta_median"] + ro["B"]["r_meta_median"]) / 2.0)

    main_table = {"base": mae(base, y), "P": mae(P, y), "TP": mae(TP, y), "const": mae(CONST, y)}

    # corrected + broken bootstraps for the three frozen comparisons
    comparisons = {
        "TP_vs_base": (TP, base),
        "TP_vs_P": (TP, P),
        "const_vs_base": (CONST, base),
    }
    boot_fixed, boot_broken = {}, {}
    for name, (pa, pb) in comparisons.items():
        boot_fixed[name] = paired_boot_fixed(pa, pb, y)
        boot_broken[name] = paired_boot_broken(pa, pb, y)

    # verification checks
    ver = verification(pbase["A"], pbase["B"], y)
    ver["const_shift_bound"] = const_bound_check(base, CONST, y, const_shift)

    # P/TP damage + amplitude
    g0 = dev_pen == 0
    damage = {
        "P": damage_stats(P, base, y),
        "TP": damage_stats(TP, base, y),
        "P_G0": damage_stats(P[g0], base[g0], y[g0]),
        "TP_G0": damage_stats(TP[g0], base[g0], y[g0]),
    }
    amplitude = {"P": amplitude_quantiles(P, base), "TP": amplitude_quantiles(TP, base)}

    # dev1792 / meta-neighbour story (fold A readout trained on meta B)
    j = 1792
    meta_B_pred = np.asarray(ro["A"]["dev_predictions_base"], np.float64)  # == F_A dev_cal
    meta_base_A = np.asarray(bases["A"]["meta_predictions_cal"], np.float64)
    meta_y_A = np.asarray(bases["A"]["meta_targets"], np.float64)
    nn = int(np.argmin(np.abs(meta_base_A - pbase["A"][j])))
    row1792 = {
        "dev_row": int(j),
        "train_index": int(dev_idx[j]),
        "penalty": int(dev_pen[j]),
        "y": float(y[j]),
        "base_avg": float(base[j]),
        "base_err": float(abs(base[j] - y[j])),
        "P_avg": float(P[j]),
        "P_err": float(abs(P[j] - y[j])),
        "base_A": float(pbase["A"][j]),
        "base_B": float(pbase["B"][j]),
        "P_A": float(pP["A"][j]),
        "P_A_correction": float(pP["A"][j] - pbase["A"][j]),
        "nearest_meta_row_in_B": {
            "meta_row": nn,
            "meta_base_A": float(meta_base_A[nn]),
            "meta_target": float(meta_y_A[nn]),
            "meta_residual": float(meta_y_A[nn] - meta_base_A[nn]),
            "abs_target_diff": float(abs(meta_y_A[nn] - y[j])),
        },
        "note": "meta nearest neighbour under p_base has a near-identical base "
                "prediction but a heterogeneous (severe) residual; the leaf-1 "
                "ExtraTrees P readout copies that residual onto the dev row.",
    }

    # severe rows listing (outer dev, penalty <= -2)
    sev = []
    for jj in np.where(dev_pen <= -2)[0].tolist():
        sev.append({"dev_row": int(jj), "train_index": int(dev_idx[jj]), "penalty": int(dev_pen[jj]),
                    "y": float(y[jj]), "base": float(base[jj]), "P": float(P[jj]), "TP": float(TP[jj]),
                    "base_err": float(abs(base[jj] - y[jj])), "P_err": float(abs(P[jj] - y[jj])),
                    "TP_err": float(abs(TP[jj] - y[jj])),
                    "base_A": float(pbase["A"][jj]), "base_B": float(pbase["B"][jj])})

    payload = {
        "source": "existing cached arrays only (no model fit, no backward)",
        "source_files": {f"F_{f}.json": str(SRC / f"F_{f}.json") for f in FOLDS} | {"readouts.json": str(SRC / "readouts.json")},
        "protocol": {"n_boot": N_BOOT, "seed": BOOT_SEED,
                     "aggregation": "average A/B predictions before MAE",
                     "positive_gain_means": "MAE(first) - MAE(second) > 0 is an improvement of first over second"},
        "main_table": main_table,
        "bootstrap_fixed": boot_fixed,
        "bootstrap_broken_original": boot_broken,
        "verification": ver,
        "damage": damage,
        "amplitude": amplitude,
        "row_1792": row1792,
        "severe_rows": sev,
        "n_dev": int(len(y)),
        "dev_strata": {"0": int((dev_pen == 0).sum()), "-1": int((dev_pen == -1).sum()),
                       "le_-2": int((dev_pen <= -2).sum())},
        "official_test_loaded": False,
    }
    (OUT / "part_a_paired_bootstrap.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "main_table": main_table,
        "fixed": {k: (round(v["point"], 8), round(v["lo"], 8), round(v["hi"], 8)) for k, v in boot_fixed.items()},
        "broken": {k: (round(v["point"], 8), round(v["lo"], 8), round(v["hi"], 8)) for k, v in boot_broken.items()},
        "ver_const": {"shift": const_shift, "row": (ver["const_shift_bound"]["row_min"], ver["const_shift_bound"]["row_max"]),
                      "boot": (ver["const_shift_bound"]["boot"]["lo"], ver["const_shift_bound"]["boot"]["hi"]),
                      "ok": ver["const_shift_bound"]["row_within_bound"] and ver["const_shift_bound"]["boot_within_bound"]},
        "P_damage": {k: damage["P"][k] for k in ("mae_gain", "n_improved", "n_worsened",
                                                  "top10_share_of_damage", "top50_share_of_damage")},
        "TP_damage": {k: damage["TP"][k] for k in ("mae_gain", "n_improved", "n_worsened",
                                                   "top10_share_of_damage", "top50_share_of_damage")},
    }, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())