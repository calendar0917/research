"""Label-blind R2 topology features, frozen Fit strata, and Dev stratum gains."""

from __future__ import annotations

from typing import Any

import numpy as np

import ib_common as C


def _feature_masks(values: np.ndarray, feature: str,
                   thresholds: dict[str, Any]) -> tuple[dict[str, np.ndarray],
                                                         list[tuple[str, str]]]:
    values = np.asarray(values, np.float64)
    if feature == "closure_22":
        masks = C.assign_count_strata(
            values, thresholds["closure_22_nonzero_median"], "closure_22")
        contrasts = [("closure_22_nz_high", "closure_22_zero"),
                     ("closure_22_nz_low", "closure_22_zero"),
                     ("closure_22_nz_high", "closure_22_nz_low")]
    elif feature == "reconvergence_2":
        masks = C.assign_count_strata(
            values, thresholds["reconvergence_2_nonzero_median"], "reconvergence_2")
        contrasts = [("reconvergence_2_nz_high", "reconvergence_2_zero"),
                     ("reconvergence_2_nz_low", "reconvergence_2_zero"),
                     ("reconvergence_2_nz_high", "reconvergence_2_nz_low")]
    elif feature == "ring_frac":
        masks = C.assign_median_strata(values, thresholds["ring_frac_median"],
                                       "ring_frac")
        if np.any(values == 0):
            masks["ring_frac_zero"] = values == 0
        contrasts = [("ring_frac_high", "ring_frac_low")]
    elif feature == "n_atoms":
        masks = C.assign_median_strata(values, thresholds["n_atoms_median"],
                                       "n_atoms")
        contrasts = [("n_atoms_high", "n_atoms_low")]
    else:
        raise KeyError(feature)
    masks = {k: v for k, v in masks.items() if np.any(v)}
    contrasts = [(a, b) for a, b in contrasts if a in masks and b in masks]
    return masks, contrasts


def main() -> None:
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    C.assert_inputs()
    bundles = {split: C.load_bundle(split) for split in C.SPLITS}
    scaffolds = {split: C.scaffold_map(split) for split in C.SPLITS}

    topology: dict[str, dict[int, dict[str, Any]]] = {}
    thresholds: dict[str, Any] = {}
    threshold_summary: dict[str, Any] = {}
    for split in C.SPLITS:
        bundle = bundles[split]
        fit_rows = np.asarray(bundle.fit_rows, np.int64)
        dev_rows = np.asarray(bundle.eval_rows, np.int64)
        fit_table = C.topology_rows(bundle, fit_rows, "fit")
        dev_table = C.topology_rows(bundle, dev_rows, "dev")
        all_rows = sorted(fit_table + dev_table, key=lambda r: int(r["row_id"]))
        C.write_csv(C.ANALYSIS_DIR / f"topology_molecules_{split}.csv", all_rows)
        topology[split] = {int(r["row_id"]): r for r in all_rows}
        th = C.fit_thresholds(fit_table)
        thresholds[split] = th

        def counts(table: list[dict[str, Any]], key: str) -> dict[str, int]:
            values = np.asarray([float(r[key]) for r in table], np.float64)
            return {
                "n": int(values.size),
                "zero": int(np.sum(values == 0)),
                "nonzero": int(np.sum(values > 0)),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
            }

        fit_medians = {}
        fit_nonzero_medians = {}
        for feat in C.ROOT_FEATURES:
            vals = np.asarray([float(r[f"{feat}_mean"]) for r in fit_table],
                              np.float64)
            fit_medians[feat] = float(np.median(vals))
            nz = vals[vals > 0]
            fit_nonzero_medians[feat] = (float(np.median(nz))
                                         if nz.size else None)
        fit_medians["ring_frac"] = float(np.median(
            [float(r["ring_frac"]) for r in fit_table]))
        fit_medians["n_atoms"] = float(np.median(
            [float(r["n_atoms"]) for r in fit_table]))
        threshold_summary[split] = {
            "thresholds": th,
            "fit_medians": fit_medians,
            "fit_nonzero_medians": fit_nonzero_medians,
            "fit": {
                "closure_22_mean": counts(fit_table, "closure_22_mean"),
                "reconvergence_2_mean": counts(fit_table, "reconvergence_2_mean"),
                "ring_frac": counts(fit_table, "ring_frac"),
                "n_atoms": counts(fit_table, "n_atoms"),
            },
            "dev": {
                "closure_22_mean": counts(dev_table, "closure_22_mean"),
                "reconvergence_2_mean": counts(dev_table, "reconvergence_2_mean"),
                "ring_frac": counts(dev_table, "ring_frac"),
                "n_atoms": counts(dev_table, "n_atoms"),
            },
        }
    C.write_json(C.ANALYSIS_DIR / "topology_thresholds.json", threshold_summary)

    strata_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    descriptive_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    all_contrasts: dict[tuple[str, int], dict[tuple[str, str], dict[str, Any]]] = {}

    for split, seed in C.CELLS:
        bundle = bundles[split]
        rows = np.asarray(bundle.eval_rows, np.int64)
        pred_r2 = C.load_predictions(C.RUNS[(split, seed)]["R2"], bundle)
        pred_ego2 = C.load_predictions(C.RUNS[(split, seed)]["EGO2"], bundle)
        y = pred_r2["y"]
        pa = pred_r2["pred"]
        pb = pred_ego2["pred"]
        clusters = C.clusters_for(rows, scaffolds[split])
        table = topology[split]
        values = {
            "closure_22": np.asarray([float(table[int(r)]["closure_22_mean"])
                                      for r in rows]),
            "reconvergence_2": np.asarray([float(table[int(r)]["reconvergence_2_mean"])
                                           for r in rows]),
            "ring_frac": np.asarray([float(table[int(r)]["ring_frac"])
                                     for r in rows]),
            "n_atoms": np.asarray([float(table[int(r)]["n_atoms"]) for r in rows]),
        }
        delta_all = np.abs(pa - y) - np.abs(pb - y)
        cell_contrasts: dict[tuple[str, str], dict[str, Any]] = {}

        for feature in C.PRIMARY_FEATURES:
            masks, contrasts = _feature_masks(values[feature], feature,
                                              thresholds[split])
            result = C.bootstrap_strata(y, pa, pb, clusters, masks, contrasts)
            for name, rec in result["strata"].items():
                mask = masks[name]
                dom = C.dominance(delta_all[mask])
                support_ok = bool(rec["n_molecules"] >= 20
                                  and rec["n_clusters"] is not None
                                  and rec["n_clusters"] >= 10)
                strata_rows.append({
                    "split": split,
                    "seed": int(seed),
                    "feature": feature,
                    "stratum": name,
                    "n_molecules": rec["n_molecules"],
                    "n_clusters": rec["n_clusters"],
                    "rmse_r2": rec["rmse_r2"],
                    "rmse_ego2": rec["rmse_ego2"],
                    "mae_r2": rec["mae_r2"],
                    "mae_ego2": rec["mae_ego2"],
                    "gain_rmse": rec["gain_rmse"],
                    "gain_rmse_ci_lo": rec["gain_rmse_ci"]["lo"],
                    "gain_rmse_ci_hi": rec["gain_rmse_ci"]["hi"],
                    "gain_mae": rec["gain_mae"],
                    "gain_mae_ci_lo": rec["gain_mae_ci"]["lo"],
                    "gain_mae_ci_hi": rec["gain_mae_ci"]["hi"],
                    "prop_ego2_better": rec["prop_ego2_better"],
                    "max_abs_share": dom["max_abs_share"],
                    "top5pct_share": dom["top5pct_share"],
                    "gain_after_max_abs": dom["gain_after_max_abs"],
                    "gain_after_top5pct": dom["gain_after_top5pct"],
                    "retained_fraction_after_top5pct": dom["retained_fraction_after_top5pct"],
                    "sign_preserved_after_max_abs": dom["sign_preserved_after_max_abs"],
                    "support_ok": support_ok,
                    "dominated_by_few": bool(dom["dominated"]),
                })
            for a, b in contrasts:
                rec = result["contrasts"][f"{a}_minus_{b}"]
                cell_contrasts[(feature, f"{a}_minus_{b}")] = rec
                n_a = result["strata"][a]["n_molecules"]
                n_b = result["strata"][b]["n_molecules"]
                cl_a = result["strata"][a]["n_clusters"]
                cl_b = result["strata"][b]["n_clusters"]
                support_ok = bool(
                    n_a >= 20 and n_b >= 20
                    and cl_a is not None and cl_b is not None
                    and cl_a >= 10 and cl_b >= 10
                )
                contrast_rows.append({
                    "split": split,
                    "seed": int(seed),
                    "feature": feature,
                    "contrast": f"{a}_minus_{b}",
                    "gain_mae": rec["gain_mae"],
                    "gain_mae_ci_lo": rec["gain_mae_ci"]["lo"],
                    "gain_mae_ci_hi": rec["gain_mae_ci"]["hi"],
                    "gain_rmse": rec["gain_rmse"],
                    "gain_rmse_ci_lo": rec["gain_rmse_ci"]["lo"],
                    "gain_rmse_ci_hi": rec["gain_rmse_ci"]["hi"],
                    "n_valid_draws": rec["n_valid_draws"],
                    "n_high": n_a,
                    "n_low": n_b,
                    "n_clusters_high": cl_a,
                    "n_clusters_low": cl_b,
                    "support_ok": support_ok,
                })

        all_contrasts[(split, seed)] = cell_contrasts

        # Descriptive table for the remaining pre-registered topology features.
        for feature in ("closure_11", "n_nodes", "n_edges", "cycle_rank",
                        "supports3", "supports4", "n_atoms"):
            key = f"{feature}_mean" if feature not in ("n_atoms",) else "n_atoms"
            if key not in table[int(rows[0])]:
                continue
            vals = np.asarray([float(table[int(r)][key]) for r in rows])
            if feature in ("n_nodes", "n_edges", "supports3", "supports4",
                           "n_atoms"):
                th = threshold_summary[split]["fit_medians"][feature]
                masks = {"low": vals <= th, "high": vals > th}
            else:
                masks = {"zero": vals == 0, "nonzero": vals > 0}
                med = threshold_summary[split]["fit_nonzero_medians"].get(feature)
                if np.any(vals > 0) and med is not None:
                    masks["nz_low"] = (vals > 0) & (vals <= med)
                    masks["nz_high"] = vals > med
            for name, mask in masks.items():
                if not np.any(mask):
                    continue
                point = C.point_metrics(y, pa, pb, mask=mask, clusters=clusters)
                point.pop("delta", None)
                descriptive_rows.append({
                    "split": split,
                    "seed": int(seed),
                    "feature": feature,
                    "stratum": name,
                    "n_molecules": point["n_molecules"],
                    "n_clusters": point["n_clusters"],
                    "gain_mae": point["gain_mae"],
                    "gain_rmse": point["gain_rmse"],
                    "prop_ego2_better": point["prop_ego2_better"],
                })

        # Descriptive atom-count control crossed with closure_22 zero/nonzero.
        c22 = values["closure_22"]
        atoms = values["n_atoms"]
        atom_med = thresholds[split]["n_atoms_median"]
        c22_zero = c22 == 0
        c22_nonzero = c22 > 0
        control_masks = {
            "closure22_zero_atoms_low": c22_zero & (atoms <= atom_med),
            "closure22_zero_atoms_high": c22_zero & (atoms > atom_med),
            "closure22_nonzero_atoms_low": c22_nonzero & (atoms <= atom_med),
            "closure22_nonzero_atoms_high": c22_nonzero & (atoms > atom_med),
        }
        control_masks = {k: v for k, v in control_masks.items() if np.any(v)}
        if control_masks:
            cres = C.bootstrap_strata(y, pa, pb, clusters, control_masks, [])
            for name, rec in cres["strata"].items():
                control_rows.append({
                    "split": split,
                    "seed": int(seed),
                    "cell": name,
                    "n_molecules": rec["n_molecules"],
                    "n_clusters": rec["n_clusters"],
                    "gain_mae": rec["gain_mae"],
                    "gain_mae_ci_lo": rec["gain_mae_ci"]["lo"],
                    "gain_mae_ci_hi": rec["gain_mae_ci"]["hi"],
                    "gain_rmse": rec["gain_rmse"],
                    "gain_rmse_ci_lo": rec["gain_rmse_ci"]["lo"],
                    "gain_rmse_ci_hi": rec["gain_rmse_ci"]["hi"],
                })

    # D0 seed consistency of the pre-specified contrasts.
    consistency_rows: list[dict[str, Any]] = []
    keys = set(all_contrasts[("D0", 0)].keys()) | set(all_contrasts[("D0", 1)].keys())
    for feature, contrast in sorted(keys):
        rec0 = all_contrasts[("D0", 0)].get((feature, contrast))
        rec1 = all_contrasts[("D0", 1)].get((feature, contrast))
        recd1 = all_contrasts[("D1", 0)].get((feature, contrast))
        if rec0 is None or rec1 is None:
            continue
        g0 = rec0["gain_mae"]
        g1 = rec1["gain_mae"]
        gd1 = recd1["gain_mae"] if recd1 is not None else None
        sign0 = int(np.sign(g0)) if g0 is not None else 0
        sign1 = int(np.sign(g1)) if g1 is not None else 0
        sign_d1 = int(np.sign(gd1)) if gd1 is not None else 0
        same = bool(sign0 != 0 and sign0 == sign1)
        strong = bool(same and abs(g0) >= 0.005 and abs(g1) >= 0.005)
        clearly_reversed = bool(
            gd1 is not None and sign_d1 != 0 and sign_d1 != sign0
            and abs(gd1) >= 0.01
        )
        consistency_rows.append({
            "feature": feature,
            "contrast": contrast,
            "gain_mae_s0": g0,
            "gain_mae_s1": g1,
            "sign_s0": sign0,
            "sign_s1": sign1,
            "same_sign_d0": same,
            "both_abs_ge_0_005": strong,
            "gain_mae_d1": gd1,
            "sign_d1": sign_d1,
            "clearly_reversed_d1": clearly_reversed,
            "not_clearly_reversed_d1": not clearly_reversed,
        })

    # Write one table per cell.
    by_cell_strata: dict[tuple[str, int], list[dict[str, Any]]] = {
        cell: [] for cell in C.CELLS
    }
    for row in strata_rows:
        by_cell_strata[(row["split"], int(row["seed"]))].append(row)
    by_cell_contrast: dict[tuple[str, int], list[dict[str, Any]]] = {
        cell: [] for cell in C.CELLS
    }
    for row in contrast_rows:
        by_cell_contrast[(row["split"], int(row["seed"]))].append(row)
    for split, seed in C.CELLS:
        C.write_csv(C.ANALYSIS_DIR / f"strata_{split}_S{seed}.csv",
                    by_cell_strata[(split, seed)])
        C.write_csv(C.ANALYSIS_DIR / f"strata_contrasts_{split}_S{seed}.csv",
                    by_cell_contrast[(split, seed)])
    C.write_csv(C.ANALYSIS_DIR / "strata_descriptive.csv", descriptive_rows)
    C.write_csv(C.ANALYSIS_DIR / "stratified_control_atomcount.csv", control_rows)
    C.write_csv(C.ANALYSIS_DIR / "strata_d0_consistency.csv", consistency_rows)

    print("R2 topology and frozen Fit strata")
    for split in C.SPLITS:
        th = threshold_summary[split]["thresholds"]
        print(f"  {split} thresholds: closure_22 nonzero median="
              f"{th['closure_22_nonzero_median']}, reconvergence_2 nonzero median="
              f"{th['reconvergence_2_nonzero_median']}, ring median="
              f"{th['ring_frac_median']}, atoms median={th['n_atoms_median']}")
        for key in ("closure_22_mean", "reconvergence_2_mean"):
            fit = threshold_summary[split]["fit"][key]
            dev = threshold_summary[split]["dev"][key]
            print(f"    {key}: Fit zero/nonzero {fit['zero']}/{fit['nonzero']}, "
                  f"Dev zero/nonzero {dev['zero']}/{dev['nonzero']}")
    print("  primary strata:")
    for row in strata_rows:
        if row["feature"] in C.PRIMARY_FEATURES:
            print(f"    {row['split']}/S{row['seed']} {row['feature']} "
                  f"{row['stratum']}: n={row['n_molecules']} "
                  f"gain_mae={row['gain_mae']:+.4f} "
                  f"CI=[{row['gain_mae_ci_lo']:+.4f},{row['gain_mae_ci_hi']:+.4f}] "
                  f"support_ok={row['support_ok']} "
                  f"dominated={row['dominated_by_few']}")
    print("  pre-specified contrasts:")
    for row in contrast_rows:
        if row["contrast"].endswith("_high_minus_" + row["feature"] + "_zero") \
                or row["contrast"] == "ring_frac_high_minus_ring_frac_low":
            print(f"    {row['split']}/S{row['seed']} {row['feature']} "
                  f"{row['contrast']}: {row['gain_mae']:+.4f} "
                  f"CI=[{row['gain_mae_ci_lo']:+.4f},{row['gain_mae_ci_hi']:+.4f}] "
                  f"support_ok={row['support_ok']}")


if __name__ == "__main__":
    main()