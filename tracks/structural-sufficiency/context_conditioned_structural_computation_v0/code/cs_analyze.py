"""Performance analysis: main table, paired comparisons, fixed stratum, references.

Section 9 of ``PROTOCOL.md`` and nothing more.  All comparisons are paired on the
same molecules (identical official row ids); Test is a DEVELOPMENT_TEST set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cs_common as C

C.setup_paths()

import cs_data as D  # noqa: E402
import cs_rules as R  # noqa: E402


def unit_dir(dataset: str, arm: str) -> Path:
    return C.RUNS_DIR / f"{dataset.replace('ogbg-', '')}_{arm}_seed{C.SEED}"


def load_unit(dataset: str, arm: str) -> dict[str, Any] | None:
    out_dir = unit_dir(dataset, arm)
    run_path = out_dir / "run.json"
    if not run_path.is_file():
        return None
    run = C.read_json(run_path)
    preds: dict[str, Any] = {}
    for tag in ("best", "final", "last20"):
        path = out_dir / f"predictions_{tag}.npz"
        if path.is_file():
            z = np.load(path)
            preds[tag] = {k: z[k] for k in z.files}
    curve = out_dir / "train_curve.csv"
    return {
        "run": run,
        "predictions": preds,
        "curve": str(curve) if curve.is_file() else None,
        "dir": C.track_rel(out_dir),
    }


def main_table(units: Mapping[tuple[str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for (dataset, arm), unit in units.items():
        run = unit["run"]
        best = run.get("best_metrics") or {}
        row = {
            "dataset": dataset,
            "arm": arm,
            "arm_name": C.ARMS[arm]["name"],
            "parameters": int(run["parameters"]),
            "n_parameters_new": int(run.get("n_parameters_new", 0)),
            "best_epoch": run.get("best_epoch"),
            "status": run.get("status"),
            "epochs_run": run.get("epochs_run"),
        }
        for split in C.SPLITS:
            for metric in ("rmse", "mae"):
                row[f"best_{split}_{metric}"] = (best.get(split) or {}).get(f"{metric}_raw")
        for split in C.SPLITS:
            for metric in ("rmse", "mae"):
                row[f"final_{split}_{metric}"] = run.get(f"final_metrics", {}).get(f"{split}_{metric}_raw")
                row[f"last20_{split}_{metric}"] = run.get("last20_mean_metrics", {}).get(f"{split}_{metric}_raw")
        row.update(
            {
                "train_seconds": run.get("train_seconds"),
                "eval_seconds": run.get("eval_seconds"),
                "total_seconds": run.get("total_seconds"),
                "gpu_hours": run.get("gpu_hours"),
                "inference_rows_per_second": run.get("inference_rows_per_second"),
                "gpu_peak_allocated_bytes": run.get("gpu_peak_allocated_bytes"),
                "train_valid_gap_best": (
                    (best.get("valid") or {}).get("rmse_raw", float("nan"))
                    - (best.get("train") or {}).get("rmse_raw", float("nan"))
                ),
            }
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# paired comparisons
# ---------------------------------------------------------------------------


def paired_bootstrap(
    pred_left: np.ndarray,
    pred_right: np.ndarray,
    y: np.ndarray,
    *,
    replicates: int = C.BOOTSTRAP_REPLICATES,
    seed: int = C.BOOTSTRAP_SEED,
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n = int(y.size)
    rmse_d, mae_d = np.empty(replicates), np.empty(replicates)
    if clusters is None:
        for r in range(replicates):
            idx = rng.integers(0, n, n)
            yy = y[idx]
            rmse_d[r] = C.rmse(pred_left[idx], yy) - C.rmse(pred_right[idx], yy)
            mae_d[r] = C.mae(pred_left[idx], yy) - C.mae(pred_right[idx], yy)
        kind = "molecule"
    else:
        uniq = np.unique(clusters)
        buckets = [np.nonzero(clusters == u)[0] for u in uniq]
        for r in range(replicates):
            pick = rng.integers(0, len(uniq), len(uniq))
            idx = np.concatenate([buckets[i] for i in pick])
            yy = y[idx]
            rmse_d[r] = C.rmse(pred_left[idx], yy) - C.rmse(pred_right[idx], yy)
            mae_d[r] = C.mae(pred_left[idx], yy) - C.mae(pred_right[idx], yy)
        kind = "scaffold_cluster"
    out = {}
    for name, draws, point in (
        ("rmse_raw", rmse_d, C.rmse(pred_left, y) - C.rmse(pred_right, y)),
        ("mae_raw", mae_d, C.mae(pred_left, y) - C.mae(pred_right, y)),
    ):
        lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
        out[name] = {
            "point": float(point),
            "ci95_low": lo,
            "ci95_high": hi,
            "contains_zero": bool(lo <= 0.0 <= hi),
            "relative_point": float(point / C.rmse(pred_right, y)) if name == "rmse_raw" else None,
        }
    out["kind"] = kind
    out["replicates"] = int(replicates)
    return out


def paired_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    arm_left: str,
    arm_right: str,
    dataset: str,
    tag: str,
    scaffolds: Mapping[str, list[str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {"dataset": dataset, "contrast": f"{arm_left}-{arm_right}", "checkpoint": tag, "splits": {}}
    for split in ("valid", "test"):
        zl = left["predictions"].get(tag)
        zr = right["predictions"].get(tag)
        if zl is None or zr is None:
            out["splits"][split] = {"error": "missing predictions"}
            continue
        rows_l = zl[f"{split}_row_ids"]
        rows_r = zr[f"{split}_row_ids"]
        if not np.array_equal(rows_l, rows_r):
            order = np.argsort(rows_r)
            rows_r = rows_r[order]
            pr = zr[f"{split}_pred_raw"][order]
            yr = zr[f"{split}_y_raw"][order]
        else:
            pr = zr[f"{split}_pred_raw"]
            yr = zr[f"{split}_y_raw"]
        pl = zl[f"{split}_pred_raw"]
        y = zl[f"{split}_y_raw"]
        if not np.allclose(y, yr):
            raise RuntimeError(f"{dataset} {split}: label mismatch between arms")
        err_l = np.abs(pl - y)
        err_r = np.abs(pr - y)
        se_l = (pl - y) ** 2
        se_r = (pr - y) ** 2
        dse = se_l - se_r
        scaf = np.asarray(scaffolds[split], dtype=object)
        block = {
            "n_molecules": int(y.size),
            "rmse_left": C.rmse(pl, y),
            "rmse_right": C.rmse(pr, y),
            "mae_left": C.mae(pl, y),
            "mae_right": C.mae(pr, y),
            "delta_rmse": C.rmse(pl, y) - C.rmse(pr, y),
            "delta_mae": C.mae(pl, y) - C.mae(pr, y),
            "relative_delta_rmse": (C.rmse(pl, y) - C.rmse(pr, y)) / C.rmse(pr, y),
            "improved_fraction": float(np.mean(err_l < err_r)),
            "degraded_fraction": float(np.mean(err_l > err_r)),
            "tied_fraction": float(np.mean(err_l == err_r)),
            "bootstrap_molecule": paired_bootstrap(pl, pr, y),
            "bootstrap_scaffold": paired_bootstrap(pl, pr, y, clusters=scaf) if len(set(scaf.tolist())) > 1 else None,
            "n_scaffolds": int(len(set(scaf.tolist()))),
        }
        # sensitivity: drop the molecules with the largest squared-error difference
        order = np.argsort(-np.abs(dse))
        drops = []
        for k in (1, 3, 5):
            keep = np.ones(y.size, dtype=bool)
            keep[order[:k]] = False
            drops.append(
                {
                    "dropped": int(k),
                    "delta_rmse": float(C.rmse(pl[keep], y[keep]) - C.rmse(pr[keep], y[keep])),
                    "direction_preserved": bool(
                        np.sign(C.rmse(pl[keep], y[keep]) - C.rmse(pr[keep], y[keep]))
                        == np.sign(block["delta_rmse"])
                    ),
                }
            )
        block["drop_top_dse"] = drops
        block["top_dse_share"] = float(np.abs(dse).max() / np.abs(dse).sum()) if dse.size else 0.0
        block["largest_error_molecule"] = {
            "row_id": int(rows_l[int(np.argmax(np.maximum(err_l, err_r)))]),
            "abs_error_left": float(err_l.max()),
            "abs_error_right": float(err_r.max()),
        }
        out["splits"][split] = block
    return out


# ---------------------------------------------------------------------------
# fixed stratum: heterogeneous relation neighbourhood fraction
# ---------------------------------------------------------------------------


def hetero_fraction(bundle, split: str) -> np.ndarray:
    """Fraction of atoms with at least two distinct real relation types, per molecule."""
    cache = bundle.caches[split]
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
    out = np.zeros(int(cache["n_molecules"]), dtype=np.float64)
    for m in range(int(cache["n_molecules"])):
        n_atoms = int(atom_ptr[m + 1] - atom_ptr[m])
        lo, hi = int(bond_ptr[m]), int(bond_ptr[m + 1])
        groups = [set() for _ in range(n_atoms)]
        for k in range(lo, hi):
            u, v = int(bond_index[k][0]), int(bond_index[k][1])
            rel = C.BOND_TYPE_TO_RELATION[int(bond_type[k])]
            groups[u].add(rel)
            groups[v].add(rel)
        out[m] = float(np.mean([len(g) >= 2 for g in groups])) if n_atoms else 0.0
    return out


def strata(bundle, units: Mapping[tuple[str, str], Mapping[str, Any]], *, threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        "definition": "per molecule: fraction of atoms with at least two distinct real relation types among incident bonds",
        "threshold_source": "Train median of that fraction (fixed in advance of looking at Valid/Test outcomes)",
        "threshold": float(threshold),
        "splits": {},
    }
    for split in ("valid", "test"):
        fracs = hetero_fraction(bundle, split)
        high = fracs >= threshold
        entry = {
            "n_molecules": int(fracs.size),
            "n_high": int(high.sum()),
            "n_low": int((~high).sum()),
            "mean_fraction": float(fracs.mean()),
            "median_fraction": float(np.median(fracs)),
        }
        for (dataset, arm), unit in units.items():
            if dataset != bundle.dataset:
                continue
            z = unit["predictions"].get("best")
            if z is None:
                continue
            entry[f"{arm}_rmse_high"] = C.rmse(z[f"{split}_pred_raw"][high], z[f"{split}_y_raw"][high]) if high.any() else None
            entry[f"{arm}_rmse_low"] = C.rmse(z[f"{split}_pred_raw"][~high], z[f"{split}_y_raw"][~high]) if (~high).any() else None
        if ("valid" in split or True):
            pass
        out["splits"][split] = entry
    # D-C and A-D contrasts inside each stratum
    for split in ("valid", "test"):
        fracs = hetero_fraction(bundle, split)
        high = fracs >= threshold
        entry = out["splits"][split]
        for left, right in (("D", "C"), ("D", "A"), ("B", "A")):
            zl = units.get((bundle.dataset, left), {}).get("predictions", {}).get("best")
            zr = units.get((bundle.dataset, right), {}).get("predictions", {}).get("best")
            if zl is None or zr is None:
                continue
            for name, mask in (("high", high), ("low", ~high)):
                if mask.sum() < 5:
                    entry[f"{left}-{right}_{name}"] = None
                    continue
                y = zl[f"{split}_y_raw"][mask]
                pl = zl[f"{split}_pred_raw"][mask]
                pr = zr[f"{split}_pred_raw"][mask]
                entry[f"{left}-{right}_{name}"] = {
                    "n": int(mask.sum()),
                    "delta_rmse": C.rmse(pl, y) - C.rmse(pr, y),
                    "delta_mae": C.mae(pl, y) - C.mae(pr, y),
                    "relative_delta_rmse": (C.rmse(pl, y) - C.rmse(pr, y)) / C.rmse(pr, y),
                }
    return out


# ---------------------------------------------------------------------------
# historical reference (descriptive only)
# ---------------------------------------------------------------------------


def historical_reference() -> dict[str, Any]:
    out: dict[str, Any] = {
        "role": "descriptive absolute reference on the same official scaffold split; not a matched control "
                "(different architecture, readout, initialisation and batch order)",
        "sources": {},
    }
    for dataset in C.DATASETS:
        entry: dict[str, Any] = {}
        for key, name in (("C1", "method_L.json"), ("GINE", "method_G.json")):
            path = C.SQL_RESULTS / dataset / name
            if not path.is_file():
                continue
            payload = C.read_json(path)
            if key == "C1":
                best = payload.get("selected") or {}
                entry[key] = {
                    "n_parameters": payload.get("n_parameters"),
                    "selected": best,
                    "final": payload.get("final"),
                    "last20": payload.get("last20"),
                    "recipe": payload.get("recipe"),
                }
            else:
                entry[key] = {
                    "n_parameters": payload.get("n_parameters"),
                    "selected": payload.get("selected"),
                    "final_round_selected": payload.get("final_round_selected"),
                    "last20_selected": payload.get("last20_selected"),
                    "selected_peak_lr": payload.get("selected_peak_lr"),
                }
        out["sources"][dataset] = entry
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def verdicts(main: Sequence[Mapping[str, Any]], contrasts: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered reading of the performance axis (section 13.1)."""
    def get(dataset, arm, key="best_valid_rmse"):
        for row in main:
            if row["dataset"] == dataset and row["arm"] == arm:
                return row.get(key)
        return None

    def rel(dataset, left, right):
        a, b = get(dataset, left), get(dataset, right)
        if a is None or b is None:
            return None
        return float((a - b) / b)

    lipo = {
        "D_minus_A_valid": rel("ogbg-mollipo", "D", "A"),
        "D_minus_C_valid": rel("ogbg-mollipo", "D", "C"),
        "C_minus_A_valid": rel("ogbg-mollipo", "C", "A"),
        "A_minus_B_valid": rel("ogbg-mollipo", "A", "B"),
        "D_minus_A_test": rel("ogbg-mollipo", "D", "A"),
        "D_minus_C_test": None,
    }
    # Test-side relatives use the same fields of the best checkpoint
    def rel_test(dataset, left, right):
        a = None
        b = None
        for row in main:
            if row["dataset"] == dataset and row["arm"] == left:
                a = row.get("best_test_rmse")
            if row["dataset"] == dataset and row["arm"] == right:
                b = row.get("best_test_rmse")
        if a is None or b is None:
            return None
        return float((a - b) / b)

    lipo["D_minus_A_test"] = rel_test("ogbg-mollipo", "D", "A")
    lipo["D_minus_C_test"] = rel_test("ogbg-mollipo", "D", "C")
    lipo["C_minus_A_test"] = rel_test("ogbg-mollipo", "C", "A")
    lipo["A_minus_B_test"] = rel_test("ogbg-mollipo", "A", "B")
    tol = 0.02
    out = {"lipo_relative_best_rmse": lipo, "practical_threshold_relative": tol, "labels": []}
    labels = out["labels"]
    d_vs_a = lipo["D_minus_A_valid"]
    d_vs_c = lipo["D_minus_C_valid"]
    c_vs_a = lipo["C_minus_A_valid"]
    if d_vs_a is not None and d_vs_c is not None and d_vs_a <= -tol and d_vs_c <= -tol:
        labels.append("RELATION_CONDITIONED_GAIN")
    elif c_vs_a is not None and c_vs_a <= -tol and (d_vs_c is None or d_vs_c > -tol):
        labels.append("GENERIC_CONDITIONING_GAIN")
    a_vs_b = lipo["A_minus_B_valid"]
    if a_vs_b is not None and a_vs_b <= -tol and not any(x in labels for x in ("RELATION_CONDITIONED_GAIN", "GENERIC_CONDITIONING_GAIN")):
        labels.append("FIXED_SOURCE_ONLY")
    if not labels:
        labels.append("NO_MATERIAL_GAIN")
    out["note"] = (
        "2% relative RMSE is a practical reference, not a significance threshold; split/uncertainty conflicts are "
        "reported next to the label, never averaged away"
    )
    return out



def interpretability_axis(performance: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-registered reading of the three interpretability dimensions (section 13.2).

    Each dimension is read from its own artefact, with the thresholds written in the
    protocol: TRACE from the decomposition closure, RULE_REUSE from the shallow trees
    and the candidate rules, STRUCTURE_DISCOVERY from the repeated-structure stability.
    """
    out: dict[str, Any] = {"dimensions": {}, "joint_attribution": None}
    trace = C.read_json(C.ANALYSIS_DIR / "decomposition_trace.json") if (C.ANALYSIS_DIR / "decomposition_trace.json").is_file() else None
    rules = C.read_json(C.ANALYSIS_DIR / "structure_rules.json") if (C.ANALYSIS_DIR / "structure_rules.json").is_file() else None

    if trace is None:
        out["dimensions"]["TRACE"] = {"label": None, "note": "decomposition_trace.json missing"}
    else:
        worst = 0.0
        over = 0
        change = 0.0
        for dataset, blk in trace["datasets"].items():
            for arm, entry in blk["arms"].items():
                for split, sp in entry["splits"].items():
                    worst = max(worst, float(sp["closure"]["max_abs_std"]))
                    over += int(sp["closure"]["n_over_tolerance"])
                    change = max(change, float(sp["closure"]["pred_change_rms_std"]))
        out["dimensions"]["TRACE"] = {
            "label": bool(over == 0 and change > 0.0),
            "max_abs_closure_std": worst,
            "n_over_tolerance": over,
            "max_pred_change_rms_std": change,
            "criterion": "attribution closes to a non-zero prediction change on every split (0 molecules over tolerance)",
        }

    if rules is None:
        out["dimensions"]["RULE_REUSE"] = {"label": None, "note": "structure_rules.json missing"}
        out["dimensions"]["STRUCTURE_DISCOVERY"] = {"label": None, "note": "structure_rules.json missing"}
    else:
        gains = []
        cand_ok = []
        for dataset, blk in rules["datasets"].items():
            for arm, entry in blk["arms"].items():
                if "error" in entry:
                    continue
                tree = entry["trees"]["skill_vs_stronger_single"]["valid"]
                gains.append(float(tree["relative_mae_improvement"]))
                for cand in entry["candidates"]["candidates"]:
                    cand_ok.append(
                        float(cand["valid"].get("coverage_fraction") or 0.0) >= 0.01
                        and float(cand["valid"].get("same_sign_fraction") or 0.0) >= 0.8
                    )
        best_gain = max(gains) if gains else 0.0
        out["dimensions"]["RULE_REUSE"] = {
            "label": bool(best_gain >= 0.10 and any(cand_ok)),
            "best_sc_relative_mae_improvement": best_gain,
            "n_candidates_supported_and_same_sign": int(sum(cand_ok)),
            "criterion": "SC beats the stronger single input set by >=10% relative MAE AND some Train-only candidate keeps its sign on >=1% of Valid messages",
        }
        stable = []
        for dataset, blk in rules["datasets"].items():
            for arm, entry in blk["arms"].items():
                if "error" in entry or "repeated_structures" not in entry:
                    continue
                summary = entry["repeated_structures"]["summary"]
                stable.append(
                    float(summary["median_sign_change_fraction"]) <= 0.5
                    and float(summary["median_effect_to_iqr_ratio"]) >= 1.0
                )
        out["dimensions"]["STRUCTURE_DISCOVERY"] = {
            "label": bool(any(stable)),
            "criterion": "a repeated exact structural record keeps its effect sign (median sign-change fraction <= 0.5) with effect size >= 1 x its within-family IQR",
            "per_arm": stable,
        }

    labels = {k: v.get("label") for k, v in out["dimensions"].items()}
    perf_labels = list(performance.get("labels") or [])
    has_perf_gain = any(x in ("RELATION_CONDITIONED_GAIN", "GENERIC_CONDITIONING_GAIN") for x in perf_labels)
    has_struct_gain = bool(labels.get("RULE_REUSE")) or bool(labels.get("STRUCTURE_DISCOVERY"))
    rel = performance.get("lipo_relative_best_rmse") or {}
    worst_new_arm = max(
        [abs(float(rel[k])) for k in ("D_minus_A_valid", "D_minus_A_test", "D_minus_C_valid", "D_minus_C_test") if k in rel]
        or [1.0]
    )
    kept = worst_new_arm <= 0.02 and not has_perf_gain
    if None in labels.values():
        joint = "INSUFFICIENT_OR_CONFLICT"
    elif has_perf_gain and has_struct_gain:
        joint = "PERFORMANCE_AND_STRUCTURE"
    elif has_perf_gain:
        joint = "PERFORMANCE_ONLY"
    elif has_struct_gain and kept:
        joint = "STRUCTURE_ONLY_KEEPING_PERFORMANCE"
    elif not has_struct_gain and not has_perf_gain:
        joint = "NEITHER"
    else:
        joint = "INSUFFICIENT_OR_CONFLICT"
    out["joint_attribution"] = {
        "label": joint,
        "reading": {
            "PERFORMANCE_AND_STRUCTURE": "性能改善 + 可归纳结构作用",
            "PERFORMANCE_ONLY": "性能改善但无可解释性额外证据",
            "STRUCTURE_ONLY_KEEPING_PERFORMANCE": "性能基本保持 + 更好可归纳结构作用",
            "NEITHER": "两者均无实质改善",
            "INSUFFICIENT_OR_CONFLICT": "证据不足或冲突（解释性提升伴随性能明显下降也不计为联合突破）",
        }[joint],
        "performance_labels": perf_labels,
        "interpretability_labels": labels,
        "worst_relative_change_of_new_arms": worst_new_arm,
        "performance_kept": bool(kept),
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args(argv)
    units: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in C.DATASETS:
        for arm in C.ARM_ORDER:
            unit = load_unit(dataset, arm)
            if unit is None:
                C.log(f"missing unit {dataset} {arm}")
                continue
            units[(dataset, arm)] = unit
    main_rows = main_table(units)
    C.write_csv(C.ANALYSIS_DIR / "main_table.csv", main_rows)

    bundles = {dataset: D.load_bundle(dataset) for dataset in C.DATASETS}
    scaffolds = {dataset: {split: R.scaffolds(bundles[dataset], split) for split in C.SPLITS} for dataset in C.DATASETS}

    comparisons: dict[str, Any] = {"round_id": C.ROUND_ID, "created_utc": C.utcnow(), "pairs": []}
    pairs = [("A", "B"), ("C", "A"), ("D", "C"), ("D", "A")]
    for dataset in C.DATASETS:
        for left, right in pairs:
            if (dataset, left) not in units or (dataset, right) not in units:
                continue
            for tag in ("best", "final", "last20"):
                comparisons["pairs"].append(
                    paired_comparison(
                        units[(dataset, left)],
                        units[(dataset, right)],
                        arm_left=left,
                        arm_right=right,
                        dataset=dataset,
                        tag=tag,
                        scaffolds=scaffolds[dataset],
                    )
                )
    # best vs late-horizon direction consistency
    consistency = []
    for entry in comparisons["pairs"]:
        if entry["checkpoint"] != "best":
            continue
        for split in ("valid", "test"):
            blk = entry["splits"].get(split, {})
            if not blk or "delta_rmse" not in blk:
                continue
            late = next(
                (
                    e["splits"][split]
                    for e in comparisons["pairs"]
                    if e["dataset"] == entry["dataset"]
                    and e["contrast"] == entry["contrast"]
                    and e["checkpoint"] == "last20"
                ),
                None,
            )
            if late and "delta_rmse" in late:
                consistency.append(
                    {
                        "dataset": entry["dataset"],
                        "contrast": entry["contrast"],
                        "split": split,
                        "delta_best": blk["delta_rmse"],
                        "delta_last20": late["delta_rmse"],
                        "same_direction": bool(np.sign(blk["delta_rmse"]) == np.sign(late["delta_rmse"])),
                    }
                )
    comparisons["best_vs_last20_direction"] = consistency
    comparisons["historical_reference"] = historical_reference()
    C.write_json(C.ANALYSIS_DIR / "paired_comparisons.json", comparisons)

    # fixed stratum
    strata_out: dict[str, Any] = {"round_id": C.ROUND_ID, "created_utc": C.utcnow(), "datasets": {}}
    for dataset in C.DATASETS:
        bundle = bundles[dataset]
        train_frac = hetero_fraction(bundle, "train")
        threshold = float(np.median(train_frac))
        strata_out["datasets"][dataset] = {
            "train_median_threshold": threshold,
            "train_mean_fraction": float(train_frac.mean()),
            "train_zero_fraction_molecules": int((train_frac == 0).sum()),
            **strata(bundle, units, threshold=threshold),
        }
    C.write_json(C.ANALYSIS_DIR / "strata.json", strata_out)

    verdict = verdicts(main_rows, comparisons)
    verdict["interpretability_axis"] = interpretability_axis(verdict)
    C.write_json(C.ANALYSIS_DIR / "verdicts.json", verdict)
    C.log("analysis written: main_table.csv, paired_comparisons.json, strata.json, verdicts.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
