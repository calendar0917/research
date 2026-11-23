"""Route C driver: frozen real-edit response kernels on D0 and D1.

The catalogue is frozen first (``sr_catalogue.py freeze``).  This script reads
labels only after verifying the freeze hash, then:

* fits ZERO / EXACT_MEAN / K_TRANSFORM / K_CONDITIONAL_R2 / K_CONTEXT_ID
  (and optionally K_CONDITIONAL_FULL) with fold-only scalars and 3-fold
  component CV;
* evaluates Fit-Fit, Dev-Dev (primary) and Fit-Dev in original label units;
* reports the pre-registered strata and a paired component bootstrap;
* runs the algebraic functional checks and the synthetic conditional control.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import sr_ckernel as K
import sr_common as C
import sr_data as D
import sr_wl as WL


def functional_checks() -> dict[str, Any]:
    rng = np.random.default_rng(20261123)
    n, m, dx, dc = 20, 8, 3, 2
    x = rng.normal(size=(n, dx))
    c = rng.normal(size=(n, dc))
    w = rng.uniform(0.5, 1.5, size=n)
    y = rng.normal(size=n)
    xm = rng.normal(size=(m, dx))
    cm = rng.normal(size=(m, dc))
    w = w / w.mean()
    Kt = x @ x.T
    Kc = c @ c.T
    Ktc = Kt * (1.0 + Kc)
    Kt_off = xm @ x.T
    Kc_off = cm @ c.T
    Ktc_off = (Kt_off * (1.0 + Kc_off))
    lam = 1e-3
    pred = K.fit_krr(Ktc, Ktc_off, y, w, lam)
    # explicit tensor-product feature ridge
    feats = np.stack([np.outer(xi, np.concatenate([[1.0], ci])).reshape(-1)
                      for xi, ci in zip(x, c)], axis=0)
    feats_off = np.stack([np.outer(xi, np.concatenate([[1.0], ci])).reshape(-1)
                          for xi, ci in zip(xm, cm)], axis=0)
    A = feats.T @ (w[:, None] * feats) + n * lam * np.eye(feats.shape[1])
    beta = np.linalg.solve(A, feats.T @ (w * y))
    explicit = feats_off @ beta
    ridge_err = float(np.abs(pred - explicit).max())
    # swap / zero tests on the real normalized kernel construction
    swap = K.fit_krr(Ktc, -Ktc_off, y, w, lam)
    swap_err = float(np.abs(swap + pred).max())
    zero = K.fit_krr(Ktc, np.zeros_like(Ktc_off), y, w, lam)
    zero_err = float(np.abs(zero).max())
    eigs = np.linalg.eigvalsh(Ktc)
    psd_err = float(max(0.0, -eigs.min()))
    # synthetic conditional positive control
    rng2 = np.random.default_rng(7331)
    xtr = rng2.normal(size=(32, 2))
    ctr = rng2.normal(size=(32, 2))
    ytr = xtr[:, 0] + 0.4 * xtr[:, 1] + 0.7 * xtr[:, 0] * ctr[:, 0] - 0.3 * xtr[:, 1] * ctr[:, 1]
    xte = rng2.normal(size=(32, 2))
    cte = rng2.normal(size=(32, 2))
    yte = xte[:, 0] + 0.4 * xte[:, 1] + 0.7 * xte[:, 0] * cte[:, 0] - 0.3 * xte[:, 1] * cte[:, 1]
    wc = np.ones(32)
    Kt_c = xtr @ xtr.T
    Kc_c = ctr @ ctr.T
    Ktc_c = Kt_c * (1.0 + Kc_c)
    Kt_c_off = xte @ xtr.T
    Ktc_c_off = (xte @ xtr.T) * (1.0 + cte @ ctr.T)
    ptc = K.fit_krr(Ktc_c, Ktc_c_off, ytr, wc, 1e-10)
    pt = K.fit_krr(Kt_c, Kt_c_off, ytr, wc, 1e-10)
    mae_tc = float(np.mean(np.abs(ptc - yte)))
    mae_t = float(np.mean(np.abs(pt - yte)))
    return {
        "krr_vs_explicit_feature_ridge_max_abs_error": ridge_err,
        "swap_sign_max_abs_error": swap_err,
        "identical_pair_max_abs_error": zero_err,
        "kernel_psd_min_eigenvalue_negative_part": psd_err,
        "conditional_positive_control": {
            "mae_ktc": mae_tc, "mae_kt": mae_t, "lambda": 1e-10,
            "pass": bool(mae_tc < 1e-4 and mae_tc < mae_t),
        },
        "pass": bool(ridge_err <= 1e-9 and swap_err <= 1e-10
                     and zero_err <= 1e-12 and psd_err <= 1e-9
                     and mae_tc < 1e-4 and mae_tc < mae_t),
    }


def _train_components(design: K.Design) -> np.ndarray:
    return K.sub_components(design.train_idx, design.arrays["row_a"],
                            design.arrays["row_b"])


def fit_scope(design: K.Design, lambdas: Mapping[str, float], scope: str,
              itr: np.ndarray, comp_train: np.ndarray, *,
              with_id: bool, with_full: bool) -> dict[str, Any]:
    iev = design.scope_rows(scope)
    w = design.weights
    sigma = float(np.std(design.bundle.labels[design.row_fit]))
    if not (sigma > 0):
        sigma = 1.0
    y = design.delta[itr] / sigma
    ks = K.build_kernels(design, itr, iev, w, with_id=with_id, with_full=with_full)
    preds: dict[str, np.ndarray] = {"ZERO": np.zeros(iev.size)}
    em, em_info = K.exact_mean_predict(design, itr, iev, comp_train)
    preds["EXACT_MEAN"] = em
    used = {}
    for name in ("K_TRANSFORM", "K_CONDITIONAL_R2", "K_CONTEXT_ID",
                 "K_CONDITIONAL_FULL"):
        if name not in lambdas:
            continue
        Ktr, Koff = K._kernel_pair(ks, name)
        if Ktr is None or Koff is None:
            continue
        preds[name] = K.fit_krr(Ktr, Koff, y, w, lambdas[name]) * sigma
        used[name] = float(lambdas[name])
    return {"idx": iev, "pred": preds, "lambdas": used, "sigma": sigma,
            "exact_mean_info": em_info, "kernel_scalars": ks.scalars}


def metrics_for(design: K.Design, idx: np.ndarray, pred: np.ndarray,
                scope: str, model: str) -> dict[str, Any]:
    err = pred - design.delta[idx]
    comp = design.global_component[idx]
    return {
        "split": design.split, "scope": scope, "model": model,
        "component_equal_mae": K.component_equal_mae(np.abs(err), comp),
        "pair_equal_mae": K.pair_equal_mae(err),
        "rmse": K.pair_rmse(err),
        "pred_std": float(np.std(pred)),
        "n_pairs": int(idx.size),
        "n_molecules": int(len(set(design.arrays["row_a"][idx].tolist())
                              | set(design.arrays["row_b"][idx].tolist()))),
        "n_components": int(len(np.unique(comp))),
        "mean_delta_y": float(np.mean(design.delta[idx])),
        "mean_pred": float(np.mean(pred)),
    }


def support_sets(design: K.Design, itr: np.ndarray) -> dict[str, set]:
    train_transforms = {str(design.transform_id[i]) for i in itr.tolist()}
    comp = _train_components(design)
    transform_components: dict[str, set[int]] = defaultdict(set)
    for i, c in zip(itr.tolist(), comp.tolist()):
        transform_components[str(design.transform_id[i])].add(int(c))
    fit_cores = {str(design.core_key[i]) for i in itr.tolist()}
    fit_set = set(int(x) for x in design.row_fit.tolist())
    legal_cores = set()
    for row, key in zip(design.arrays["cand_row"].tolist(),
                        design.arrays["cand_c_key"].astype(str).tolist()):
        if int(row) in fit_set:
            legal_cores.add(str(key))
    train_global_components = set(design.global_component[itr].tolist())
    return {"transforms": train_transforms,
            "transform_components": transform_components,
            "cores": fit_cores,
            "legal_cores": legal_cores,
            "global_components": train_global_components}


def strata_rows(design: K.Design, idx: np.ndarray, preds: Mapping[str, np.ndarray],
                support: Mapping[str, set], scope: str) -> list[dict[str, Any]]:
    rows = []
    transform = design.transform_id[idx]
    core = design.core_key[idx]
    comp = design.global_component[idx]
    ncomp = np.asarray([len(support["transform_components"].get(str(t), ()))
                        for t in transform.tolist()], np.int64)
    strict_core = np.asarray([str(c) not in support["legal_cores"] for c in core.tolist()])
    seen_core = np.asarray([str(c) in support["cores"] for c in core.tolist()])
    has_support = np.asarray([int(c) in support["global_components"] for c in comp.tolist()])
    strata = {
        "transform": np.asarray(["seen" if str(t) in support["transforms"] else "unseen"
                                 for t in transform.tolist()]),
        "support_components": np.asarray(["0" if n == 0 else ("1" if n == 1 else "2+")
                                          for n in ncomp.tolist()]),
        "core_seen_in_fit_pairs": np.asarray(["seen" if s else "unseen"
                                              for s in seen_core.tolist()]),
        "core_seen_in_any_legal_fit_cut": np.asarray(["seen" if not s else "strict_new"
                                                      for s in strict_core.tolist()]),
        "eval_component_has_fit_support": np.asarray(["yes" if s else "no"
                                                      for s in has_support.tolist()]),
    }
    for stratum, values in strata.items():
        for value in sorted(set(values.tolist())):
            mask = values == value
            if not mask.any():
                continue
            block_comp = comp[mask]
            for model, pred in preds.items():
                err = np.abs(pred[mask] - design.delta[idx[mask]])
                rows.append({
                    "split": design.split, "scope": scope, "stratum": stratum,
                    "value": value, "model": model,
                    "n_pairs": int(mask.sum()),
                    "n_components": int(len(np.unique(block_comp))),
                    "component_equal_mae": K.component_equal_mae(err, block_comp),
                    "pair_equal_mae": K.pair_equal_mae(err),
                    "exact_transform_coverage": float(np.mean(
                        [str(t) in support["transforms"] for t in transform[mask].tolist()]))
                    if mask.any() else None,
                })
    return rows


def bootstrap(design: K.Design, idx: np.ndarray,
              preds: Mapping[str, np.ndarray], scope: str,
              n_boot: int = 5000, seed: int = 1703) -> dict[str, Any]:
    comp = design.global_component
    all_components = np.unique(comp).tolist()
    # positions inside the scope, grouped by their global component
    by_comp: dict[int, np.ndarray] = {}
    for c in np.unique(comp[idx]).tolist():
        by_comp[int(c)] = np.flatnonzero(comp[idx] == c)
    rng = np.random.default_rng(int(seed))
    models = list(preds)
    out = {m: {"component_equal_mae": np.zeros(n_boot),
               "pair_equal_mae": np.zeros(n_boot)} for m in models}
    n_comp = len(all_components)
    for b in range(n_boot):
        draw = rng.integers(0, n_comp, size=n_comp)
        blocks = [by_comp.get(int(all_components[j])) for j in draw.tolist()]
        blocks = [block for block in blocks if block is not None and block.size]
        if not blocks:
            continue
        pos = np.concatenate(blocks)
        for m in models:
            err = np.abs(preds[m][pos] - design.delta[idx[pos]])
            per_block = np.asarray([float(np.mean(np.abs(
                preds[m][block] - design.delta[idx[block]]))) for block in blocks])
            out[m]["component_equal_mae"][b] = float(per_block.mean())
            out[m]["pair_equal_mae"][b] = float(np.mean(err))
    summary = {}
    for m in models:
        summary[m] = {}
        for metric, values in out[m].items():
            lo, hi = np.percentile(values, [2.5, 97.5])
            summary[m][metric] = {"mean": float(values.mean()),
                                  "ci95": [float(lo), float(hi)]}
    return {"scope": scope, "split": design.split, "n_boot": int(n_boot),
            "seed": int(seed), "n_scope_pairs": int(idx.size),
            "n_scope_components": int(len(by_comp)),
            "n_global_components": int(n_comp),
            "resample_unit": "official-Train pair-graph component",
            "summary": summary}


def run_split(split: str, *, with_full: bool, n_boot: int) -> dict[str, Any]:
    t0 = time.time()
    design = K.build_design(split, want_full=with_full)
    cv = K.cv_select(design, with_id=True, with_full=with_full)
    comp_train = _train_components(design)
    support = support_sets(design, design.train_idx)
    lambdas = dict(cv.get("lambda", {}))
    results: dict[str, Any] = {"cv": {k: v for k, v in cv.items()
                                     if k != "oof_errors"},
                               "design": {
                                   "n_pairs_total": int(design.arrays["row_a"].size),
                                   "n_train_pairs": int(design.train_idx.size),
                                   "n_dev_pairs": int(design.dev_idx.size),
                                   "n_fitdev_pairs": int(design.fitdev_idx.size),
                                   "n_global_components": int(len(np.unique(
                                       design.global_component))),
                                   "sigma_fit": float(np.std(
                                       design.bundle.labels[design.row_fit])),
                               },
                               "scopes": {}, "metrics": [], "strata": [],
                               "bootstrap": {}, "support": {}}
    for scope in ("Fit-Fit", "Dev-Dev", "Fit-Dev"):
        idx = design.scope_rows(scope)
        fit = fit_scope(design, lambdas, scope, design.train_idx, comp_train,
                        with_id=True, with_full=with_full)
        preds = fit["pred"]
        for model, pred in preds.items():
            results["metrics"].append(metrics_for(design, idx, pred, scope, model))
        results["scopes"][scope] = {
            "n_pairs": int(idx.size),
            "n_molecules": int(len(set(design.arrays["row_a"][idx].tolist())
                                 | set(design.arrays["row_b"][idx].tolist()))),
            "n_components": int(len(np.unique(design.global_component[idx]))),
            "n_full_cores": int(len(set(design.core_key[idx].tolist()))),
            "n_transforms": int(len(set(design.transform_id[idx].tolist()))),
            "exact_mean": fit["exact_mean_info"],
            "lambdas": fit["lambdas"],
        }
        if scope == "Dev-Dev":
            results["strata"] = strata_rows(design, idx, preds, support, scope)
            if n_boot:
                results["bootstrap"]["Dev-Dev"] = bootstrap(
                    design, idx, preds, scope, n_boot=n_boot)
        if scope == "Fit-Fit" and n_boot:
            results["bootstrap"]["Fit-Fit"] = bootstrap(
                design, idx, preds, scope, n_boot=max(500, n_boot // 5))
    # support table
    train_transforms = support["transforms"]
    train_components = support["transform_components"]
    fit_cores = support["cores"]
    strict_legal = support["legal_cores"]
    for scope in ("Fit-Fit", "Dev-Dev", "Fit-Dev"):
        idx = design.scope_rows(scope)
        tr = design.transform_id[idx].astype(str)
        core = design.core_key[idx].astype(str)
        ncomp = np.asarray([len(train_components.get(t, ())) for t in tr.tolist()])
        results["support"][scope] = {
            "pairs": int(idx.size),
            "molecules": int(len(set(design.arrays["row_a"][idx].tolist())
                               | set(design.arrays["row_b"][idx].tolist()))),
            "full_cores": int(len(set(core.tolist()))),
            "exact_transforms": int(len(set(tr.tolist()))),
            "transform_support_counts": {
                "unseen": int((ncomp == 0).sum()),
                "one_component": int((ncomp == 1).sum()),
                "two_plus_components": int((ncomp >= 2).sum()),
            },
            "core_seen_in_fit_pairs": int(sum(str(c) in fit_cores for c in core.tolist())),
            "core_seen_in_any_legal_fit_cut": int(sum(str(c) in strict_legal
                                                      for c in core.tolist())),
            "pair_graph_components": int(len(np.unique(
                design.global_component[idx]))),
        }
    results["seconds"] = time.time() - t0
    results["label_hashes"] = {
        "labels_sha256": design.bundle.hashes["train_labels"]["sha256"],
        "split_identity": D.split_identity(design.bundle)["identity_sha256"],
        "freeze_sha256": design.freeze_meta["pairs_sha256"],
    }
    return results


def write_results(payload: Mapping[str, Any]) -> None:
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = []
    strata = []
    for split, block in payload["splits"].items():
        metrics.extend(block["metrics"])
        strata.extend(block["strata"])
    C.write_csv(C.ANALYSIS_DIR / "response_metrics.csv", metrics)
    C.write_csv(C.ANALYSIS_DIR / "response_strata.csv", strata)
    C.write_json(C.ANALYSIS_DIR / "response_support.json", payload.get("support", {}))
    C.write_json(C.ANALYSIS_DIR / "response_bootstrap.json",
                 {s: b["bootstrap"] for s, b in payload["splits"].items()})
    C.write_json(C.ANALYSIS_DIR / "response_cv.json",
                 {s: b["cv"] for s, b in payload["splits"].items()})
    C.write_json(C.ANALYSIS_DIR / "response_checks.json", payload["checks"])
    C.write_json(C.ANALYSIS_DIR / "response_run.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="D0")
    parser.add_argument("--full", action="store_true",
                        help="also run K_CONDITIONAL_FULL (first optional CPU model)")
    parser.add_argument("--boot", type=int, default=5000)
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "response_run.json"))
    args = parser.parse_args()
    C.assert_pins()
    checks = functional_checks()
    payload: dict[str, Any] = {"protocol_id": "structure-route-resolution-8h-v1",
                               "generated_utc": C.now_utc(), "checks": checks,
                               "splits": {}, "support": {},
                               "software": {"blas_threads": 1}}
    for split in args.splits.split(","):
        block = run_split(split, with_full=bool(args.full), n_boot=int(args.boot))
        payload["splits"][split] = block
        payload["support"][split] = block["support"]
    write_results(payload)
    print(json.dumps({"checks_pass": checks["pass"],
                      "splits": {s: {"cv_lambda": b["cv"].get("lambda"),
                                     "dev": next((m for m in b["metrics"]
                                                  if m["scope"] == "Dev-Dev"),
                                                 None)}
                                 for s, b in payload["splits"].items()}},
                     indent=1, default=str)[:3000])
    return 0 if checks["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())