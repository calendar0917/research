"""SES v0 numerical correction and multi-candidate complementarity diagnosis.

Two independent analyses, both label-free for the aggregation methods:

1. correct the frozen-selection intervention: the parent code used the
   *RANDOM* arm's expected MSE as the SELECT arm's uniform-expected baseline;
2. describe what several frozen SELECT/RANDOM candidates can and cannot buy.

The candidate oracle uses labels only as a post-hoc diagnostic.  It is never
used to choose an aggregation, a model or a checkpoint.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

import sed_common as C

SPLITS = ("valid", "test")
N_BOOT = 1000
BOOT_SEED = 0


def _load_inputs() -> dict[str, Any]:
    path = C.SES_RESULTS / "per_molecule_candidates.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _arrays(z: dict[str, Any], split: str, arm: str) -> dict[str, np.ndarray]:
    y = np.asarray(z[f"{split}_y_raw"], np.float64)
    ptr = np.asarray(z[f"{split}_cand_ptr"], np.int64)
    pred = np.asarray(z[f"{split}_cand_pred_{arm}"], np.float64)
    return {"y": y, "ptr": ptr, "pred": pred}


def _segment_expectation(ptr: np.ndarray, pred: np.ndarray, y: np.ndarray,
                         power: int) -> np.ndarray:
    """Per-molecule mean of |candidate prediction - y|**power."""
    out = np.empty(y.size, np.float64)
    for i in range(y.size):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        out[i] = np.mean(np.abs(pred[lo:hi] - y[i]) ** power)
    return out


def _det_metrics(pred: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    err = pred - y
    return err * err, np.abs(err)


def _expected_pair_mse(ptr: np.ndarray, pred: np.ndarray, y: np.ndarray,
                       *, mae: bool = False) -> np.ndarray:
    """Exact expectation over a uniformly random unordered pair of candidates."""
    out = np.empty(y.size, np.float64)
    for i in range(y.size):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        p = pred[lo:hi]
        vals = []
        for a in range(p.size):
            for b in range(a + 1, p.size):
                avg = 0.5 * (p[a] + p[b])
                vals.append(abs(avg - y[i]) if mae else (avg - y[i]) ** 2)
        out[i] = float(np.mean(vals)) if vals else (
            abs(p[0] - y[i]) if mae else (p[0] - y[i]) ** 2)
    return out


def _aggregate_summary(se: np.ndarray, ae: np.ndarray) -> dict[str, float]:
    return {"rmse": C.rmse(se), "mae": C.mae_from_ae(ae)}


def _dispersion(ptr: np.ndarray, pred: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    n = y.size
    exp_mse = np.empty(n, np.float64)
    var = np.empty(n, np.float64)
    mean_se = np.empty(n, np.float64)
    for i in range(n):
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        p = pred[lo:hi]
        mu = float(p.mean())
        exp_mse[i] = float(np.mean((p - y[i]) ** 2))
        var[i] = float(np.var(p, ddof=0))
        mean_se[i] = (mu - y[i]) ** 2
    identity = exp_mse - (mean_se + var)
    good = exp_mse > 1e-12
    ratio = np.full(n, np.nan, np.float64)
    ratio[good] = var[good] / exp_mse[good]
    return {
        "identity_max_abs_error": float(np.max(np.abs(identity))),
        "mean_candidate_variance": float(np.mean(var)),
        "mean_expected_mse": float(np.mean(exp_mse)),
        "mean_variance_fraction": float(np.nanmean(ratio)),
        "median_variance_fraction": float(np.nanmedian(ratio)),
        "pooled_variance_fraction": float(var.sum() / max(exp_mse.sum(), 1e-12)),
    }


def _candidate_sets(z: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nptr = np.asarray(z[f"{split}_cand_node_ptr"], np.int64)
    nodes = np.asarray(z[f"{split}_cand_nodes"], np.int64)
    bptr = np.asarray(z[f"{split}_cand_bond_ptr"], np.int64)
    bonds = np.asarray(z[f"{split}_cand_bonds"], np.int64)
    return nptr, nodes, bptr, bonds


def _coverage_one(ptr: np.ndarray, idx: np.ndarray, nptr: np.ndarray,
                  nodes: np.ndarray, bptr: np.ndarray, bonds: np.ndarray,
                  orig_n: np.ndarray, orig_m: np.ndarray) -> dict[str, np.ndarray]:
    n = ptr.size - 1
    node_cov = np.empty(n, np.float64)
    edge_cov = np.empty(n, np.float64)
    for i in range(n):
        ns: set[int] = set()
        bs: set[int] = set()
        lo, hi = int(ptr[i]), int(ptr[i + 1])
        for j in (range(lo, hi) if idx is None else np.atleast_1d(idx[i]).tolist()):
            a0, a1 = int(nptr[j]), int(nptr[j + 1])
            b0, b1 = int(bptr[j]), int(bptr[j + 1])
            ns.update(int(x) for x in nodes[a0:a1].tolist())
            bs.update(int(x) for x in bonds[b0:b1].tolist())
        node_cov[i] = len(ns) / float(max(1, int(orig_n[i])))
        edge_cov[i] = len(bs) / float(max(1, int(orig_m[i])))
    return {"node": node_cov, "edge": edge_cov}


def _cov_summary(v: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "min": float(np.min(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


def run() -> dict[str, Any]:
    t0 = time.perf_counter()
    z = _load_inputs()
    n_boot = N_BOOT
    boot_seed = BOOT_SEED

    intervention_rows: list[dict[str, Any]] = []
    aggregation_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    dispersion_rows: list[dict[str, Any]] = []

    summary: dict[str, Any] = {
        "input": str((C.SES_RESULTS / "per_molecule_candidates.npz").relative_to(C.REPO_ROOT)),
        "input_sha256": C.sha256_file(C.SES_RESULTS / "per_molecule_candidates.npz"),
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "splits": {},
    }

    # ------------------------------------------------------------------
    # 1. frozen-selection intervention (SELECT predictor)
    # ------------------------------------------------------------------
    erratum: dict[str, Any] = {}
    for split in SPLITS:
        sel = _arrays(z, split, "select")
        rnd = _arrays(z, split, "random")
        y, ptr, pred_sel = sel["y"], sel["ptr"], sel["pred"]
        scores = np.asarray(z[f"{split}_cand_scores_select"], np.float64)
        chosen = np.asarray(z[f"{split}_chosen_index"], np.int64)
        pred = np.asarray(z[f"{split}_pred_select"], np.float64)
        # sanity: the stored argmax prediction is the chosen candidate
        chosen_pred = np.array([pred_sel[int(ptr[i]) + int(chosen[i])]
                                for i in range(y.size)], np.float64)
        if not np.allclose(chosen_pred, pred, atol=1e-12):
            raise RuntimeError(f"{split}: stored SELECT prediction is not the argmax candidate")

        # A: selector argmax
        se_a, ae_a = _det_metrics(pred, y)
        # B_correct: SELECT's own uniform expected MSE
        mse_bc = _segment_expectation(ptr, pred_sel, y, 2)
        mae_bc = _segment_expectation(ptr, pred_sel, y, 1)
        # B_wrong (parent): RANDOM's uniform expected MSE
        mse_bw = _segment_expectation(ptr, rnd["pred"], y, 2)
        mae_bw = _segment_expectation(ptr, rnd["pred"], y, 1)
        # C: minimum selector score
        se_c = np.empty(y.size, np.float64)
        ae_c = np.empty(y.size, np.float64)
        for i in range(y.size):
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            j = lo + int(np.argmin(scores[lo:hi]))
            se_c[i] = (pred_sel[j] - y[i]) ** 2
            ae_c[i] = abs(pred_sel[j] - y[i])

        contrast = {
            "A_vs_B_correct": C.paired_bootstrap(se_a, mse_bc, n_boot=n_boot, seed=boot_seed),
            "A_vs_B_erratum_random": C.paired_bootstrap(se_a, mse_bw, n_boot=n_boot, seed=boot_seed),
            "A_vs_C": C.paired_bootstrap(se_a, se_c, n_boot=n_boot, seed=boot_seed),
        }
        manifest = C.read_json(C.SES_RESULTS / "select" / "manifest.json")
        manifest_expected = float(manifest["best_metrics"][split]["rmse_expected"])
        manifest_argmax = float(manifest["best_metrics"][split]["rmse"])
        erratum[split] = {
            "n": int(y.size),
            "manifest_rmse_expected": manifest_expected,
            "manifest_rmse_expected_abs_diff": abs(C.rmse(mse_bc) - manifest_expected),
            "manifest_rmse_argmax": manifest_argmax,
            "manifest_rmse_argmax_abs_diff": abs(C.rmse(se_a) - manifest_argmax),
            "A_argmax": _aggregate_summary(se_a, ae_a),
            "B_uniform_select_correct": {"rmse": C.rmse(mse_bc), "mae": C.mae_from_ae(mae_bc)},
            "B_uniform_random_erratum": {"rmse": C.rmse(mse_bw), "mae": C.mae_from_ae(mae_bw)},
            "C_min_score": {"rmse": C.rmse(se_c), "mae": C.mae_from_ae(ae_c)},
            "contrasts": contrast,
            "parent_reported_B_rmse": 1.03981957811456 if split == "valid" else 0.983728935105715,
            "parent_reported_A_minus_B": -0.1299855 if split == "valid" else -0.0694699,
        }
        for variant, se, ae, note in (
            ("A_argmax", se_a, ae_a, "SELECT selector argmax"),
            ("B_uniform_select_correct", mse_bc, mae_bc,
             "SELECT own per-candidate mean squared/absolute error"),
            ("B_uniform_random_erratum", mse_bw, mae_bw,
             "parent code used RANDOM's expected MSE (erratum)"),
            ("C_min_score", se_c, ae_c, "minimum selector score"),
        ):
            intervention_rows.append({
                "split": split, "variant": variant, "n": int(y.size),
                "rmse": C.rmse(se), "mae": C.mae_from_ae(ae), "note": note,
            })
        for name, rec in contrast.items():
            contrast_rows.append({
                "split": split, "analysis": "selection_intervention", "contrast": name,
                "delta_rmse": rec["delta_a_minus_b"], "ci_lo": rec["delta_ci95"][0],
                "ci_hi": rec["delta_ci95"][1], "n_boot": n_boot,
            })
        # exact corrected MAE differences (not bootstrap): useful for the report
        erratum[split]["mae_A_minus_B_correct"] = float(np.mean(ae_a) - np.mean(mae_bc))
        erratum[split]["mae_A_minus_B_erratum"] = float(np.mean(ae_a) - np.mean(mae_bw))

    summary["erratum"] = {
        "description": (
            "The parent analysis used per_split['random_expected_mse'] as the "
            "SELECT-uniform-expected baseline in the frozen-selection intervention. "
            "The corrected baseline is the SELECT arm's own per-candidate mean "
            "squared/absolute error. The corrected RMSE must equal the SELECT "
            "manifest rmse_expected."
        ),
        "parent_reported_valid_A_minus_B": -0.1300,
        "parent_reported_test_A_minus_B": -0.0695,
        "splits": erratum,
        "unaffected": (
            "The main-table SELECT-minus-RANDOM comparison already used RANDOM's "
            "expected MSE on both sides and is unaffected. The route label was "
            "based on that comparison, so it is unaffected."
        ),
    }

    # ------------------------------------------------------------------
    # 2. multi-candidate aggregation / complementarity
    # ------------------------------------------------------------------
    candidate_summary: dict[str, Any] = {}
    for split in SPLITS:
        candidate_summary[split] = {}
        dense_pred = np.asarray(z[f"{split}_dense_pred"], np.float64)
        y_all = np.asarray(z[f"{split}_y_raw"], np.float64)
        dense_se, dense_ae = _det_metrics(dense_pred, y_all)
        aggregation_rows.append({
            "split": split, "arm": "DENSE", "method": "full_graph",
            "n": int(y_all.size), "rmse": C.rmse(dense_se),
            "mae": C.mae_from_ae(dense_ae), "note": "reference",
        })
        candidate_summary[split]["DENSE"] = _aggregate_summary(dense_se, dense_ae)

        nptr, nodes, bptr, bonds = _candidate_sets(z, split)
        orig_n = np.asarray(z[f"{split}_orig_n_atoms"], np.int64)
        orig_m = np.asarray(z[f"{split}_orig_n_bonds"], np.int64)

        for arm in ("select", "random"):
            arr = _arrays(z, split, arm)
            y, ptr, pred = arr["y"], arr["ptr"], arr["pred"]
            if not np.array_equal(y, y_all):
                raise RuntimeError("split label order mismatch")
            n = y.size
            k = np.diff(ptr)
            # exact methods
            mse_uniform = _segment_expectation(ptr, pred, y, 2)
            mae_uniform = _segment_expectation(ptr, pred, y, 1)
            avg_pred = np.array([pred[int(ptr[i]):int(ptr[i + 1])].mean()
                                 for i in range(n)], np.float64)
            med_pred = np.array([np.median(pred[int(ptr[i]):int(ptr[i + 1])])
                                 for i in range(n)], np.float64)
            se_avg, ae_avg = _det_metrics(avg_pred, y)
            se_med, ae_med = _det_metrics(med_pred, y)
            # oracle: post-hoc, uses labels
            se_orc = np.empty(n, np.float64)
            ae_orc = np.empty(n, np.float64)
            for i in range(n):
                lo, hi = int(ptr[i]), int(ptr[i + 1])
                e = pred[lo:hi] - y[i]
                se_orc[i] = float(np.min(e * e))
                ae_orc[i] = float(np.min(np.abs(e)))
            mse_r2 = _expected_pair_mse(ptr, pred, y, mae=False)
            mae_r2 = _expected_pair_mse(ptr, pred, y, mae=True)

            methods = {
                "uniform_expected": (mse_uniform, mae_uniform,
                                     "mean over single candidates (expected)"),
                "all_candidate_mean": (se_avg, ae_avg, "equal-weight mean of all candidates"),
                "all_candidate_median": (se_med, ae_med, "median of all candidates"),
                "candidate_oracle": (se_orc, ae_orc, "post-hoc label oracle; not deployable"),
                "random_two_expected": (mse_r2, mae_r2,
                                        "exact expectation over two distinct candidates"),
            }
            arm_summary: dict[str, Any] = {
                "n_molecules": int(n),
                "candidates_per_molecule_mean": float(k.mean()),
                "candidates_per_molecule_min": int(k.min()),
                "candidates_per_molecule_max": int(k.max()),
                "methods": {},
            }
            for name, (se, ae, note) in methods.items():
                arm_summary["methods"][name] = _aggregate_summary(se, ae)
                aggregation_rows.append({
                    "split": split, "arm": arm.upper(), "method": name,
                    "n": int(n), "rmse": C.rmse(se), "mae": C.mae_from_ae(ae), "note": note,
                })

            if arm == "select":
                scores = np.asarray(z[f"{split}_cand_scores_select"], np.float64)
                top1 = np.empty(n, np.int64)
                top2_pred = np.empty(n, np.float64)
                for i in range(n):
                    lo, hi = int(ptr[i]), int(ptr[i + 1])
                    order = np.argsort(-scores[lo:hi], kind="stable")
                    top1[i] = lo + int(order[0])
                    top2_pred[i] = 0.5 * (pred[lo + int(order[0])] + pred[lo + int(order[1])])
                se_top1 = (pred[top1] - y) ** 2
                ae_top1 = np.abs(pred[top1] - y)
                se_top2, ae_top2 = _det_metrics(top2_pred, y)
                arm_summary["methods"]["top1_select"] = _aggregate_summary(se_top1, ae_top1)
                arm_summary["methods"]["top2_select"] = _aggregate_summary(se_top2, ae_top2)
                aggregation_rows.append({
                    "split": split, "arm": arm.upper(), "method": "top1_select",
                    "n": int(n), "rmse": C.rmse(se_top1), "mae": C.mae_from_ae(ae_top1),
                    "note": "highest selector score",
                })
                aggregation_rows.append({
                    "split": split, "arm": arm.upper(), "method": "top2_select",
                    "n": int(n), "rmse": C.rmse(se_top2), "mae": C.mae_from_ae(ae_top2),
                    "note": "mean of the two highest selector scores",
                })
                for name, a, b in (
                    ("top2_minus_top1", se_top2, se_top1),
                    ("top2_minus_random_two_expected", se_top2, mse_r2),
                    ("all_candidate_mean_minus_uniform", se_avg, mse_uniform),
                    ("all_candidate_mean_minus_dense", se_avg, dense_se),
                    ("all_candidate_mean_minus_top1", se_avg, se_top1),
                    ("all_candidate_median_minus_mean", se_med, se_avg),
                ):
                    rec = C.paired_bootstrap(a, b, n_boot=n_boot, seed=boot_seed)
                    contrast_rows.append({
                        "split": split, "analysis": "candidate_aggregation",
                        "contrast": name, "delta_rmse": rec["delta_a_minus_b"],
                        "ci_lo": rec["delta_ci95"][0], "ci_hi": rec["delta_ci95"][1],
                        "n_boot": n_boot,
                    })
                    arm_summary.setdefault("contrasts", {})[name] = rec
                # top1/top2 and all-candidate coverage
                top2_idx = np.empty(n, dtype=object)
                for i in range(n):
                    lo, hi = int(ptr[i]), int(ptr[i + 1])
                    order = np.argsort(-scores[lo:hi], kind="stable")
                    top2_idx[i] = np.array([lo + int(order[0]), lo + int(order[1])], np.int64)
                top1_idx = np.array([[int(j)] for j in top1], dtype=object)
                cov_top1 = _coverage_one(ptr, top1_idx, nptr, nodes, bptr, bonds, orig_n, orig_m)
                cov_top2 = _coverage_one(ptr, top2_idx, nptr, nodes, bptr, bonds, orig_n, orig_m)
                cov_all = _coverage_one(ptr, None, nptr, nodes, bptr, bonds, orig_n, orig_m)
                arm_summary["coverage"] = {
                    "top1_select": {"node": _cov_summary(cov_top1["node"]),
                                    "edge": _cov_summary(cov_top1["edge"])},
                    "top2_select": {"node": _cov_summary(cov_top2["node"]),
                                    "edge": _cov_summary(cov_top2["edge"])},
                    "all_candidates": {"node": _cov_summary(cov_all["node"]),
                                       "edge": _cov_summary(cov_all["edge"])},
                }
                for set_name, cov in (("top1_select", cov_top1),
                                      ("top2_select", cov_top2),
                                      ("all_candidates", cov_all)):
                    for kind in ("node", "edge"):
                        s = _cov_summary(cov[kind])
                        coverage_rows.append({
                            "split": split, "arm": "SELECT", "candidate_set": set_name,
                            "kind": kind, "n_molecules": int(n),
                            "mean": s["mean"], "median": s["median"], "min": s["min"],
                            "p10": s["p10"], "p90": s["p90"],
                        })
            else:
                cov_all = _coverage_one(ptr, None, nptr, nodes, bptr, bonds, orig_n, orig_m)
                arm_summary["coverage"] = {
                    "all_candidates": {"node": _cov_summary(cov_all["node"]),
                                       "edge": _cov_summary(cov_all["edge"])}
                }
                for kind in ("node", "edge"):
                    s = _cov_summary(cov_all[kind])
                    coverage_rows.append({
                        "split": split, "arm": "RANDOM", "candidate_set": "all_candidates",
                        "kind": kind, "n_molecules": int(n),
                        "mean": s["mean"], "median": s["median"], "min": s["min"],
                        "p10": s["p10"], "p90": s["p90"],
                    })

            disp = _dispersion(ptr, pred, y)
            arm_summary["dispersion"] = disp
            dispersion_rows.append({"split": split, "arm": arm.upper(), **disp})
            candidate_summary[split][arm.upper()] = arm_summary

        candidate_summary[split]["DENSE"] = _aggregate_summary(dense_se, dense_ae)

    summary["candidate_diagnostics"] = candidate_summary
    summary["artifacts"] = {
        "intervention_csv": "analysis/ses_intervention_correction.csv",
        "aggregation_csv": "analysis/ses_candidate_aggregation.csv",
        "contrast_csv": "analysis/ses_candidate_contrasts.csv",
        "coverage_csv": "analysis/ses_candidate_coverage.csv",
        "dispersion_csv": "analysis/ses_candidate_dispersion.csv",
    }
    C.write_csv(C.ANALYSIS_DIR / "ses_intervention_correction.csv", intervention_rows)
    C.write_csv(C.ANALYSIS_DIR / "ses_candidate_aggregation.csv", aggregation_rows)
    C.write_csv(C.ANALYSIS_DIR / "ses_candidate_contrasts.csv", contrast_rows)
    C.write_csv(C.ANALYSIS_DIR / "ses_candidate_coverage.csv", coverage_rows)
    C.write_csv(C.ANALYSIS_DIR / "ses_candidate_dispersion.csv", dispersion_rows)
    summary["seconds"] = float(time.perf_counter() - t0)
    summary["peak_rss_gb"] = C.peak_rss_gb()
    return summary


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
