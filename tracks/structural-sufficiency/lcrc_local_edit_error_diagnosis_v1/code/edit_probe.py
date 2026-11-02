"""P2/P3 — the pre-registered local-edit diagnosis probe, its freeze and the
one-shot Audit scoring.

Implements PROTOCOL.md §6–§7 of ``lcrc-local-edit-error-diagnosis-v1`` exactly:

Target (raw logD, per pair) ``e_delta = d_pred - d_y``.

Methods (all share one transformation-class prior ``b_t``):
    NATIVE      e_hat = 0
    FAMILY      e_hat = b_t
    GLOBAL      e_hat = b_t + a1*d_pred + a2*d_nheavy + a3*d_nhetero
    TRACE       e_hat = b_t + c1*dL_var_raw + c2*dL_core_raw + c3*dR_raw
    LOCAL_ONLY  e_hat = b_t + c1*dL_var_raw + c2*dL_core_raw          (secondary)

* component-equal weights: every "pair component" carries the same total weight,
  pairs inside a component share it equally; the loss is the sum of the per
  component MSEs (equivalently a weighted squared error with w_i = 1/(C*n_c)).
* weighted ridge, **no intercept**, fixed ``L2 = 5`` for every scaled coefficient
  and every class term.  No lambda search.
* continuous differences are divided by the component-equal RMS computed on the
  fitting part only (odd symmetry preserved, no mean shift); a zero-RMS column is
  set to 0 and reported.
* ``BEST_BASE`` is chosen on Diagnose by leave-one-pair-component-out CV of the
  component-equal MAE over {NATIVE, FAMILY, GLOBAL}, tie order NATIVE > FAMILY >
  GLOBAL.  TRACE is the only trace candidate.
* class terms are recomputed inside every CV fold; a class that only occurs in the
  held-out component gets a zero term for that fold.  In Audit, a class unseen in
  Diagnose gets ``b_t = 0`` (recorded, never re-fitted).
* ``--stage fit`` writes ``analysis/probe_freeze.json`` and only then may
  ``--stage audit`` read the Audit labels, once.
* ``--stage report`` adds the paired component bootstrap (2000 draws, one shared
  resample for every method and seed), the influence checks, the closure check
  and the descriptive ``gauge_stability`` audit.  Nothing here refits a model or
  a probe, and nothing here changes the pre-registered verdict rule.

The probe coefficients are *not* a physical decomposition of ``e_delta``; they
are a fitted predictor of the local edit response error.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import edit_common as C
import edit_data as ED
import edit_trace as ET

TRACE_CSV = C.ANALYSIS / "pair_traces.csv"
PROBE_FREEZE = C.ANALYSIS / "probe_freeze.json"
AUDIT_SCORES = C.ANALYSIS / "probe_audit_scores.json"


# ---------------------------------------------------------------------------
# data assembly
# ---------------------------------------------------------------------------


def load_traces() -> dict[tuple[str, int, str], dict[str, Any]]:
    if not TRACE_CSV.exists():
        raise SystemExit(f"{C.track_rel(TRACE_CSV)} is missing; run code/edit_trace.py first")
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    with open(TRACE_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["model"], int(row["seed"]), row["pair_id"])
            rec: dict[str, Any] = {}
            for k, v in row.items():
                if k in ("pair_id", "partition", "model"):
                    rec[k] = v
                else:
                    try:
                        rec[k] = float(v)
                    except (TypeError, ValueError):
                        rec[k] = v
            out[key] = rec
    return out


def load_pairs() -> list[dict[str, Any]]:
    pairs = ET.load_pairs()
    comp = C.read_json(C.ANALYSIS / "component_split.json")
    row_to_component = comp["row_to_component"]
    for p in pairs:
        cid = row_to_component.get(str(p["row_a"]))
        if cid is None:
            cid = row_to_component.get(str(p["row_b"]))
        if cid is None:
            raise SystemExit(f"{p['pair_id']}: no component id for row_a/row_b")
        p["component"] = str(cid)
    return pairs


def _row_structure(partitions: Sequence[str]) -> dict[int, tuple[int, int]]:
    """(n heavy atoms, n non-carbon heavy atoms) per row id; label-free lookup."""
    spec = ED.frozen_spec()
    carbon = {i for i, nm in enumerate(spec.atom_type_names) if nm.startswith("C")}
    out: dict[int, tuple[int, int]] = {}
    for part in partitions:
        cache = ED.load_cache(part)
        atom_ptr = np.asarray(cache["atom_ptr"])
        atom_type = np.asarray(cache["atom_type"])
        rows = np.asarray(cache["rows"]).tolist()
        for pos, row in enumerate(rows):
            a0, a1 = int(atom_ptr[pos]), int(atom_ptr[pos + 1])
            types = atom_type[a0:a1].tolist()
            out[int(row)] = (int(a1 - a0), int(sum(1 for t in types if int(t) not in carbon)))
    return out


def build_dataset(model: str, seed: int, partition: str, traces, pairs, y_by_row, structure) -> dict[str, Any]:
    rows = []
    for p in pairs:
        if p["partition"] != partition:
            continue
        tr = traces.get((model, seed, p["pair_id"]))
        if tr is None:
            raise SystemExit(f"missing trace for {model}/s{seed}/{p['pair_id']}")
        y_a = y_by_row[int(p["row_a"])]
        y_b = y_by_row[int(p["row_b"])]
        d_y = y_b - y_a
        e_delta = float(tr["d_pred_raw"]) - d_y
        rows.append(
            {
                "pair_id": p["pair_id"],
                "component": p["component"],
                "class_id": str(p["class_id"]),
                "core_class": int(p["core_class"]),
                "same_var_heavy": int(p["same_var_heavy"]),
                "row_a": int(p["row_a"]),
                "row_b": int(p["row_b"]),
                "d_pred": float(tr["d_pred_raw"]),
                "dL_var": float(tr["dL_var_raw"]),
                "dL_core": float(tr["dL_core_raw"]),
                "dR": float(tr["dR_raw"]),
                "d_nheavy": float(structure[int(p["row_b"])][0] - structure[int(p["row_a"])][0]),
                "d_nhetero": float(structure[int(p["row_b"])][1] - structure[int(p["row_a"])][1]),
                "d_y": float(d_y),
                "e_delta": float(e_delta),
            }
        )
    if not rows:
        raise SystemExit(f"no pairs for {model}/s{seed}/{partition}")
    return {
        "rows": rows,
        "components": np.asarray([r["component"] for r in rows], dtype=object),
        "X_cont": {
            "GLOBAL": np.asarray([[r["d_pred"], r["d_nheavy"], r["d_nhetero"]] for r in rows], dtype=np.float64),
            "TRACE": np.asarray([[r["dL_var"], r["dL_core"], r["dR"]] for r in rows], dtype=np.float64),
            "LOCAL_ONLY": np.asarray([[r["dL_var"], r["dL_core"]] for r in rows], dtype=np.float64),
        },
        "y": np.asarray([r["e_delta"] for r in rows], dtype=np.float64),
        "classes": np.asarray([r["class_id"] for r in rows], dtype=object),
    }


def component_weights(components: np.ndarray) -> np.ndarray:
    """w_i = 1/(C * n_c): every component has the same total weight."""
    uniq, inv, counts = np.unique(components, return_inverse=True, return_counts=True)
    n_c = counts[inv].astype(np.float64)
    return 1.0 / (float(uniq.size) * n_c)


def component_equal_mae(errors: np.ndarray, components: np.ndarray) -> float:
    uniq = np.unique(components)
    vals = []
    for c in uniq:
        m = components == c
        vals.append(float(np.mean(np.abs(errors[m]))))
    return float(np.mean(vals)) if vals else float("nan")


def component_rms(x: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Component-equal RMS of every column (no mean shift)."""
    uniq = np.unique(components)
    out = np.zeros(x.shape[1], dtype=np.float64)
    if x.size == 0:
        return out
    per = np.zeros((uniq.size, x.shape[1]), dtype=np.float64)
    for j, c in enumerate(uniq):
        m = components == c
        per[j] = np.mean(x[m] ** 2, axis=0)
    out = np.sqrt(per.mean(axis=0))
    return out


def effective_rank(x: np.ndarray) -> int:
    if x.size == 0:
        return 0
    sv = np.linalg.svd(x, compute_uv=False)
    tol = max(x.shape) * (sv[0] if sv.size else 0.0) * np.finfo(float).eps
    return int((sv > tol).sum())


# ---------------------------------------------------------------------------
# probe fitting
# ---------------------------------------------------------------------------


def _design(
    cont: np.ndarray | None,
    rms: np.ndarray | None,
    classes: np.ndarray,
    class_ids: Sequence[int],
) -> np.ndarray:
    n = classes.size
    cols = []
    if class_ids:
        onehot = np.zeros((n, len(class_ids)), dtype=np.float64)
        index = {c: i for i, c in enumerate(class_ids)}
        for i, c in enumerate(classes.tolist()):
            j = index.get(str(c))
            if j is not None:
                onehot[i, j] = 1.0
        cols.append(onehot)
    if cont is not None and rms is not None:
        scaled = np.zeros_like(cont)
        for j in range(cont.shape[1]):
            if rms[j] > 0:
                scaled[:, j] = cont[:, j] / rms[j]
        cols.append(scaled)
    if not cols:
        return np.zeros((n, 0), dtype=np.float64)
    return np.concatenate(cols, axis=1)


def fit_ridge(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float) -> np.ndarray:
    if x.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)
    xtw = x.T * w
    a = xtw @ x + float(l2) * np.eye(x.shape[1], dtype=np.float64)
    b = xtw @ y
    return np.linalg.solve(a, b)


def method_spec(method: str) -> str | None:
    return {
        "NATIVE": None,
        "FAMILY": None,
        "GLOBAL": "GLOBAL",
        "TRACE": "TRACE",
        "LOCAL_ONLY": "LOCAL_ONLY",
    }[method]


def predict(
    method: str,
    cont: np.ndarray | None,
    rms: np.ndarray | None,
    classes: np.ndarray,
    class_ids: Sequence[int],
    beta: np.ndarray,
) -> np.ndarray:
    if method == "NATIVE":
        return np.zeros(classes.size, dtype=np.float64)
    block = method_spec(method)
    x = _design(cont if block else None, rms if block else None, classes, class_ids)
    if x.shape[1] != beta.size:
        raise RuntimeError(f"{method}: design width {x.shape[1]} != coefficient count {beta.size}")
    return x @ beta


def fit_method(
    method: str,
    ds: Mapping[str, Any],
    idx: np.ndarray,
    l2: float = C.PROBE_L2,
) -> dict[str, Any]:
    block = method_spec(method)
    classes = ds["classes"][idx]
    class_ids = [] if method == "NATIVE" else sorted(set(classes.tolist()))
    cont = ds["X_cont"][block][idx] if block else None
    rms = component_rms(cont, ds["components"][idx]) if block else None
    x = _design(cont, rms, classes, class_ids)
    w = component_weights(ds["components"][idx])
    beta = fit_ridge(x, ds["y"][idx], w, l2)
    return {
        "method": method,
        "block": block,
        "class_ids": [str(c) for c in class_ids],
        "rms": None if rms is None else [float(v) for v in rms],
        "beta": [float(v) for v in beta],
        "n_fit_pairs": int(idx.size),
        "n_fit_components": int(np.unique(ds["components"][idx]).size),
        "effective_rank": int(effective_rank(x)),
        "l2": float(l2),
    }


def apply_frozen(frozen: Mapping[str, Any], ds: Mapping[str, Any], idx: np.ndarray) -> np.ndarray:
    method = frozen["method"]
    if method == "NATIVE":
        return np.zeros(idx.size, dtype=np.float64)
    block = frozen["block"]
    cont = ds["X_cont"][block][idx] if block else None
    rms = np.asarray(frozen["rms"], dtype=np.float64)
    classes = ds["classes"][idx]
    beta = np.asarray(frozen["beta"], dtype=np.float64)
    return predict(method, cont, rms, classes, frozen["class_ids"], beta)


def leave_one_component_out(
    method: str, ds: Mapping[str, Any], l2: float = C.PROBE_L2
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    components = ds["components"]
    uniq = sorted(set(components.tolist()))
    preds = np.zeros(len(ds["rows"]), dtype=np.float64)
    folds = []
    for held in uniq:
        test_mask = components == held
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        frozen = fit_method(method, ds, train_idx, l2)
        preds[test_idx] = apply_frozen(frozen, ds, test_idx)
        folds.append(
            {
                "held_component": str(held),
                "n_train_pairs": int(train_idx.size),
                "n_test_pairs": int(test_idx.size),
                "classes_in_train": int(len(frozen["class_ids"])),
                "classes_zeroed_in_fold": int(
                    len(set(ds["classes"][test_idx].tolist()) - set(frozen["class_ids"]))
                ),
            }
        )
    return preds, folds


# ---------------------------------------------------------------------------
# stage: fit
# ---------------------------------------------------------------------------


def stage_fit(models: Sequence[str], seeds: Sequence[int]) -> dict[str, Any]:
    if ED.audit_labels_accessed():
        raise SystemExit("Audit labels were already read; the round is over, refusing to refit")
    pairs = load_pairs()
    traces = load_traces()
    structure = _row_structure(["diagnose", "audit"])
    y_diag = ED.labels("diagnose", "probe_fit")
    y_by_row = {int(r): float(v) for r, v in zip(y_diag["rows"], y_diag["y"])}

    payload: dict[str, Any] = {
        "schema": "lcrc-local-edit-probe-freeze-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "code_hashes": C.code_hashes(),
        "probe_l2": C.PROBE_L2,
        "methods": list(C.PROBE_METHODS),
        "baselines": list(C.PROBE_BASELINES),
        "main_method": C.PROBE_MAIN,
        "weighting": "component-equal; w_i = 1/(C*n_component); loss = sum_c MSE_c",
        "scaling": "diagnose component-equal RMS, no mean shift, zero-RMS column -> 0",
        "missing_class_policy": "class unseen in the fitting part -> b_t = 0 (never refitted)",
        "effective_rank_note": "reported per method and per fold; a rank deficiency is shrinkage, not an error",
        "fit_partition": "diagnose",
        "select_partition_role": "checkpoint selection only; never used by the probe",
        "units": {},
        "audit_labels_accessed": False,
    }
    for model in models:
        for seed in seeds:
            ds = build_dataset(model, seed, "diagnose", traces, pairs, y_by_row, structure)
            cv: dict[str, Any] = {}
            for method in C.PROBE_METHODS:
                preds, folds = leave_one_component_out(method, ds)
                err = ds["y"] - preds
                cv[method] = {
                    "component_equal_weight_mae": component_equal_mae(err, ds["components"]),
                    "n_pairs": int(len(ds["rows"])),
                    "n_components": int(np.unique(ds["components"]).size),
                    "folds": folds,
                }
            # BEST_BASE: lowest CV MAE, ties in the frozen order NATIVE > FAMILY > GLOBAL
            order = {"NATIVE": 0, "FAMILY": 1, "GLOBAL": 2}
            best = min(
                C.PROBE_BASELINES,
                key=lambda m: (round(cv[m]["component_equal_weight_mae"], 12), order[m]),
            )
            final = {m: fit_method(m, ds, np.arange(len(ds["rows"]))) for m in C.PROBE_METHODS}
            for m in C.PROBE_METHODS:
                final[m]["diagnose_component_equal_weight_mae"] = component_equal_mae(
                    ds["y"] - apply_frozen(final[m], ds, np.arange(len(ds["rows"]))), ds["components"]
                )
            key = f"{model}:s{seed}"
            payload["units"][key] = {
                "model": model,
                "seed": int(seed),
                "n_pairs": int(len(ds["rows"])),
                "n_components": int(np.unique(ds["components"]).size),
                "cv": cv,
                "best_base": best,
                "gain_cv_trace_minus_best_base": float(
                    cv[best]["component_equal_weight_mae"] - cv["TRACE"]["component_equal_weight_mae"]
                ),
                "probes": final,
            }
            C.log(
                f"[fit] {key}: BEST_BASE={best} cvMAE base={cv[best]['component_equal_weight_mae']:.4f} "
                f"trace={cv['TRACE']['component_equal_weight_mae']:.4f} "
                f"local_only={cv['LOCAL_ONLY']['component_equal_weight_mae']:.4f}"
            )
    payload["diagnose_class_coverage"] = {
        "n_classes": len({str(p["class_id"]) for p in pairs if p["partition"] == "diagnose"}),
    }
    C.write_json(PROBE_FREEZE, payload)
    C.log(f"probe frozen -> {C.track_rel(PROBE_FREEZE)}")
    return payload


# ---------------------------------------------------------------------------
# stage: audit (one shot)
# ---------------------------------------------------------------------------


def _load_frozen() -> dict[str, Any]:
    if not PROBE_FREEZE.exists():
        raise SystemExit("analysis/probe_freeze.json is missing; run --stage fit first")
    return C.read_json(PROBE_FREEZE)


def stage_audit() -> dict[str, Any]:
    frozen = _load_frozen()
    if AUDIT_SCORES.exists():
        raise SystemExit("Audit was already scored once; refusing a second read")
    pairs = load_pairs()
    traces = load_traces()
    structure = _row_structure(["audit", "diagnose"])
    y_audit = ED.labels("audit", "audit_score")
    y_by_row = {int(r): float(v) for r, v in zip(y_audit["rows"], y_audit["y"])}

    payload: dict[str, Any] = {
        "schema": "lcrc-local-edit-probe-audit-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "probe_freeze_sha256": C.sha256_file(PROBE_FREEZE),
        "audit_label_event": y_audit["event"],
        "units": {},
    }
    for key, unit in frozen["units"].items():
        model, seed = unit["model"], int(unit["seed"])
        ds = build_dataset(model, seed, "audit", traces, pairs, y_by_row, structure)
        idx = np.arange(len(ds["rows"]))
        scores: dict[str, Any] = {}
        for method in C.PROBE_METHODS:
            probe = unit["probes"][method]
            pred = apply_frozen(probe, ds, idx)
            err = ds["y"] - pred
            unseen = sorted(set(ds["classes"].tolist()) - set(probe["class_ids"]))
            scores[method] = {
                "component_equal_weight_mae": component_equal_mae(err, ds["components"]),
                "n_pairs": int(idx.size),
                "n_components": int(np.unique(ds["components"]).size),
                "n_pairs_in_unseen_classes": int(np.isin(ds["classes"], unseen).sum()),
            }
        best = unit["best_base"]
        gain = scores[best]["component_equal_weight_mae"] - scores["TRACE"]["component_equal_weight_mae"]
        # per-pair residuals are kept for the bootstrap and the influence checks
        residuals = {}
        for method in C.PROBE_METHODS:
            residuals[method] = (ds["y"] - apply_frozen(unit["probes"][method], ds, idx)).tolist()
        payload["units"][key] = {
            "model": model,
            "seed": int(seed),
            "best_base": best,
            "scores": scores,
            "gain_trace_minus_best_base": float(gain),
            "see_unseen_classes": {
                "seen": sorted(set(ds["classes"].tolist()) & set(unit["probes"]["TRACE"]["class_ids"])),
                "unseen": sorted(set(ds["classes"].tolist()) - set(unit["probes"]["TRACE"]["class_ids"])),
            },
            "audit_pairs": [
                {
                    "pair_id": r["pair_id"],
                    "component": r["component"],
                    "class_id": r["class_id"],
                    "core_class": r["core_class"],
                    "same_var_heavy": r["same_var_heavy"],
                    "row_a": r["row_a"],
                    "row_b": r["row_b"],
                    "e_delta": r["e_delta"],
                    "d_pred": r["d_pred"],
                    "d_nheavy": r["d_nheavy"],
                }
                for r in ds["rows"]
            ],
            "residuals": residuals,
        }
        C.log(
            f"[audit] {key}: best={best} MAE_base={scores[best]['component_equal_weight_mae']:.4f} "
            f"MAE_trace={scores['TRACE']['component_equal_weight_mae']:.4f} gain={gain:+.4f}"
        )
    C.write_json(AUDIT_SCORES, payload)
    C.log(f"audit scored once -> {C.track_rel(AUDIT_SCORES)}")
    return payload


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------


def _component_mae_from_residuals(errors: np.ndarray, comps: np.ndarray) -> float:
    return component_equal_mae(errors, comps)


def stage_report(n_boot: int = C.BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    frozen = _load_frozen()
    if not AUDIT_SCORES.exists():
        raise SystemExit("analysis/probe_audit_scores.json is missing; run --stage audit first")
    audit = C.read_json(AUDIT_SCORES)
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)

    per_seed: list[dict[str, Any]] = []
    comps_by_unit: dict[str, np.ndarray] = {}
    err_by_unit: dict[str, dict[str, np.ndarray]] = {}
    for key, unit in audit["units"].items():
        pairs_meta = unit["audit_pairs"]
        comps = np.asarray([p["component"] for p in pairs_meta], dtype=object)
        comps_by_unit[key] = comps
        err_by_unit[key] = {
            m: np.asarray(unit["residuals"][m], dtype=np.float64) for m in C.PROBE_METHODS
        }
        per_seed.append(
            {
                "unit": key,
                "model": unit["model"],
                "seed": int(unit["seed"]),
                "best_base": unit["best_base"],
                "audit_mae": {m: component_equal_mae(err_by_unit[key][m], comps) for m in C.PROBE_METHODS},
                "gain_trace_minus_best_base": float(unit["gain_trace_minus_best_base"]),
            }
        )

    # paired bootstrap over Audit components; one shared resample for every
    # method and model/seed (the primary comparison is on C1, the reference on A0)
    main_units = [u for u in per_seed if u["model"] == C.MODEL_MAIN]
    shared_keys = [u["unit"] for u in main_units]
    component_set = sorted(set().union(*[set(comps_by_unit[k].tolist()) for k in shared_keys])) if shared_keys else []
    boot_gain = {k: [] for k in shared_keys}
    boot_mae = {k: {m: [] for m in C.PROBE_METHODS} for k in shared_keys}
    for _ in range(int(n_boot)):
        draw = rng.choice(len(component_set), size=len(component_set), replace=True)
        drawn = [component_set[i] for i in draw]
        for k in shared_keys:
            meta = audit["units"][k]
            comps = comps_by_unit[k]
            best = meta["best_base"]
            mask = np.isin(comps, drawn)
            err = err_by_unit[k]
            mae = {m: component_equal_mae(err[m][mask], comps[mask]) for m in C.PROBE_METHODS}
            for m in C.PROBE_METHODS:
                boot_mae[k][m].append(mae[m])
            boot_gain[k].append(mae[best] - mae["TRACE"])

    gains = [u["gain_trace_minus_best_base"] for u in main_units]
    mean_gain = float(np.mean(gains)) if gains else float("nan")
    positive = int(sum(1 for g in gains if g > 0))
    pooled = np.asarray([v for k in shared_keys for v in boot_gain[k]], dtype=np.float64)
    ci = {
        "lo": float(np.percentile(pooled, 2.5)) if pooled.size else float("nan"),
        "hi": float(np.percentile(pooled, 97.5)) if pooled.size else float("nan"),
    }

    # influence: drop the worst native component, then the worst native molecule
    influence: list[dict[str, Any]] = []
    for u in main_units:
        k = u["unit"]
        meta = audit["units"][k]
        comps = comps_by_unit[k]
        native = err_by_unit[k]["NATIVE"]
        comp_mean = {c: float(np.mean(np.abs(native[comps == c]))) for c in sorted(set(comps.tolist()))}
        worst_comp = max(comp_mean, key=lambda c: comp_mean[c])
        keep = comps != worst_comp
        without_comp = {
            m: component_equal_mae(err_by_unit[k][m][keep], comps[keep]) for m in C.PROBE_METHODS
        }
        meta_pairs = meta["audit_pairs"]
        native_by_row: dict[int, float] = {}
        for i, p in enumerate(meta_pairs):
            for r in (p["row_a"], p["row_b"]):
                native_by_row[r] = native_by_row.get(r, 0.0) + abs(float(native[i]))
        worst_row = max(native_by_row, key=lambda r: native_by_row[r])
        keep_rows = np.asarray([not (p["row_a"] == worst_row or p["row_b"] == worst_row) for p in meta_pairs])
        without_mol = {
            m: component_equal_mae(err_by_unit[k][m][keep_rows], comps[keep_rows]) for m in C.PROBE_METHODS
        }
        influence.append(
            {
                "unit": k,
                "worst_component": str(worst_comp),
                "n_pairs_in_worst_component": int((comps == worst_comp).sum()),
                "native_component_mae_worst": comp_mean[worst_comp],
                "gain_all": u["gain_trace_minus_best_base"],
                "gain_without_worst_component": float(without_comp[meta["best_base"]] - without_comp["TRACE"]),
                "worst_molecule_row": int(worst_row),
                "gain_without_worst_molecule": float(without_mol[meta["best_base"]] - without_mol["TRACE"]),
                "n_pairs_remaining_after_molecule_drop": int(keep_rows.sum()),
            }
        )
    max_single_component_share = []
    for u in main_units:
        k = u["unit"]
        comps = comps_by_unit[k]
        g = abs(u["gain_trace_minus_best_base"])
        contrib = []
        for c in sorted(set(comps.tolist())):
            m = comps == c
            meta = audit["units"][k]
            best = meta["best_base"]
            contrib.append(abs(component_equal_mae(err_by_unit[k][best][m], comps[m]) - component_equal_mae(err_by_unit[k]["TRACE"][m], comps[m])) * m.sum())
        total = sum(contrib) if contrib else 0.0
        max_single_component_share.append(
            {
                "unit": k,
                "largest_component_share_of_abs_gain": float(max(contrib) / total) if total > 0 else None,
                "n_components": len(contrib),
            }
        )

    # closure: sum of e_hat around fundamental cycles of the pair graph
    closure = _closure_check(audit)

    # descriptive gauge stability: shift branch attributions by the equivalent
    # reduction and re-score with the *frozen* coefficients (no refit)
    gauge = _gauge_stability(frozen, audit, pairs=None, traces=load_traces(), n_boot=int(n_boot))

    verdict, reason = _verdict(mean_gain, positive, len(gains), ci, max_single_component_share)

    payload = {
        "schema": "lcrc-local-edit-probe-report-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "bootstrap": {
            "seed": C.BOOTSTRAP_SEED,
            "replicates": int(n_boot),
            "unit": "audit pair component",
            "shared_resample_across_methods": True,
            "pooled_gain_ci": ci,
            "note": "CI is conditional on these 6 trained units; seed dispersion is reported separately",
        },
        "per_seed_primary": per_seed,
        "primary_model": C.MODEL_MAIN,
        "mean_gain_primary": mean_gain,
        "n_positive_seeds": positive,
        "n_seeds": len(gains),
        "gain_threshold": C.PROMISING_MIN_GAIN,
        "positive_seed_threshold": C.PROMISING_MIN_POSITIVE_SEEDS,
        "influence": influence,
        "max_single_component_share": max_single_component_share,
        "closure": closure,
        "gauge_stability": gauge,
        "verdict": verdict,
        "verdict_reason": reason,
        "audit_size": {
            "n_pairs_primary": int(len(audit["units"][main_units[0]["unit"]]["audit_pairs"])) if main_units else 0,
            "n_components_primary": int(len(set(comps_by_unit[main_units[0]["unit"]].tolist()))) if main_units else 0,
            "n_independent_components": len(component_set),
        },
    }
    C.write_json(C.ANALYSIS / "probe_report.json", payload)
    C.log(f"verdict = {verdict} ({reason})")
    return payload


def _verdict(mean_gain, positive, n_seeds, ci, shares) -> tuple[str, str]:
    if not np.isfinite(mean_gain):
        return "EXECUTION_INCOMPLETE", "no primary unit produced an Audit gain"
    biggest = None
    for s in shares:
        v = s.get("largest_component_share_of_abs_gain")
        if v is not None:
            biggest = v if biggest is None else max(biggest, v)
    if mean_gain <= 0 or mean_gain < C.PROMISING_MIN_GAIN:
        if mean_gain <= 0:
            return "NO_INCREMENT_THIS_PROBE", f"mean Gain {mean_gain:+.4f} <= 0: the trace probe is not better than the frozen baseline"
        return (
            "NO_INCREMENT_THIS_PROBE",
            f"mean Gain {mean_gain:+.4f} is below the pre-registered {C.PROMISING_MIN_GAIN} logD threshold",
        )
    ci_ok = ci["lo"] > 0.0
    seeds_ok = positive >= C.PROMISING_MIN_POSITIVE_SEEDS
    not_single = biggest is None or biggest < 0.5
    if ci_ok and seeds_ok and not_single:
        return "PROMISING_DIAGNOSTIC_INTERFACE", "all pre-registered conditions met"
    parts = []
    if not ci_ok:
        parts.append(f"component bootstrap CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}] includes 0")
    if not seeds_ok:
        parts.append(f"only {positive}/{n_seeds} seeds positive")
    if not not_single:
        parts.append(f"a single component carries {biggest:.0%} of the absolute gain")
    return "PROMISING_BUT_UNRESOLVED", "; ".join(parts)


def _closure_check(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Sum of e_hat around cycles of the pair graph over molecules (rows)."""
    unit = audit["units"].get(f"{C.MODEL_MAIN}:s0")
    if unit is None:
        return {"status": "UNAVAILABLE", "reason": "no C1 seed-0 unit"}
    pairs = unit["audit_pairs"]
    residuals = unit["residuals"]["TRACE"]
    n = len(pairs)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in pairs:
        union(p["row_a"], p["row_b"])
    # a cycle exists iff the edge count in some component exceeds its spanning tree
    from collections import defaultdict

    edges = defaultdict(int)
    nodes = defaultdict(set)
    for p in pairs:
        r = find(p["row_a"])
        edges[r] += 1
        nodes[r].add(p["row_a"])
        nodes[r].add(p["row_b"])
    cycles = {r: edges[r] - (len(nodes[r]) - 1) for r in edges}
    n_cycles = int(sum(v for v in cycles.values() if v > 0))
    if n_cycles == 0:
        return {
            "status": "UNAVAILABLE",
            "reason": "the Audit pair graph is a forest (no cycle), so a closed-loop residual cannot be formed",
            "n_edges": n,
            "n_rows": len(parent),
        }
    # triangle check on the full row graph
    adj = defaultdict(set)
    for p in pairs:
        adj[p["row_a"]].add(p["row_b"])
        adj[p["row_b"]].add(p["row_a"])
    triangles = []
    for a in adj:
        for b in adj[a]:
            if b <= a:
                continue
            common = adj[a] & adj[b]
            for c in common:
                if c > b:
                    triangles.append((a, b, c))
    if not triangles:
        return {
            "status": "UNAVAILABLE",
            "reason": "cycles exist but no triangle is available; the protocol asks for triangles or basic cycles",
            "n_edges": n,
            "n_cycles": n_cycles,
        }
    resid = []
    for (a, b, c) in triangles:
        edges_abc = []
        for (x, y) in ((a, b), (b, c), (c, a)):
            found = None
            for i, p in enumerate(pairs):
                if {p["row_a"], p["row_b"]} == {x, y}:
                    found = float(residuals[i])
                    sign = 1.0 if (p["row_a"], p["row_b"]) == (x, y) else -1.0
                    break
            if found is None:
                edges_abc = None
                break
            edges_abc.append(sign * found)
        if edges_abc is not None:
            resid.append(float(sum(edges_abc)))
    if not resid:
        return {"status": "UNAVAILABLE", "reason": "no triangle has all three pair edges present"}
    return {
        "status": "OK",
        "n_triangles": len(resid),
        "mean_abs_closure_residual": float(np.mean(np.abs(resid))),
        "max_abs_closure_residual": float(np.max(np.abs(resid))),
        "note": "a non-zero loop does not invalidate the error-prediction experiment, but it forbids claiming the single-molecule predictor is fixed",
    }


def _gauge_stability(frozen, audit, pairs, traces, n_boot: int) -> dict[str, Any]:
    """Descriptive only: shift branch attributions by an equivalent gauge move."""
    k = float(C.GAUGE_K)
    out: dict[str, Any] = {
        "note": "descriptive audit; no probe is refitted and no main metric changes",
        "gauge_k_raw_logD": k,
        "shift": "c'_v = c_v + k, rho' = rho - k*n_atoms  =>  dL_var' = dL_var + k*(n_var_B - n_var_A), dL_core' = dL_core + k*(n_core_B - n_core_A), dR' = dR - k*(n_atoms_B - n_atoms_A); d_pred unchanged",
        "units": {},
    }
    all_pairs = {p["pair_id"]: p for p in ET.load_pairs()}
    for key, unit in audit["units"].items():
        if unit["model"] != C.MODEL_MAIN:
            continue
        meta_pairs = unit["audit_pairs"]
        comps = np.asarray([p["component"] for p in meta_pairs], dtype=object)
        base_err = np.asarray(unit["residuals"]["TRACE"], dtype=np.float64)
        # rebuild gauge-shifted continuous features for the audit rows
        dL_var_s, dL_core_s, dR_s = [], [], []
        same, diff = [], []
        for p_meta in meta_pairs:
            pid = p_meta["pair_id"]
            tr = traces.get((unit["model"], int(unit["seed"]), pid))
            if tr is None:
                raise SystemExit(f"missing trace for {pid}")
            nva, nvb = float(tr["n_var_atoms_a"]), float(tr["n_var_atoms_b"])
            nca, ncb = float(tr["n_core_atoms"]), float(tr["n_core_atoms"])
            na, nb = float(tr["n_atoms_a"]), float(tr["n_atoms_b"])
            dL_var_s.append(float(tr["dL_var_raw"]) + k * (nvb - nva))
            dL_core_s.append(float(tr["dL_core_raw"]) + k * (ncb - nca))
            dR_s.append(float(tr["dR_raw"]) - k * (nb - na))
            same.append(bool(int(p_meta["same_var_heavy"])))
        dL_var_s = np.asarray(dL_var_s)
        dL_core_s = np.asarray(dL_core_s)
        dR_s = np.asarray(dR_s)
        d_pred_check = dL_var_s + dL_core_s + dR_s
        probe = frozen["units"][key]["probes"]["TRACE"]
        rms = np.asarray(probe["rms"], dtype=np.float64)
        class_ids = probe["class_ids"]
        beta = np.asarray(probe["beta"], dtype=np.float64)
        classes = np.asarray([p["class_id"] for p in meta_pairs], dtype=object)
        y = np.asarray([p["e_delta"] for p in meta_pairs], dtype=np.float64)
        shifted = np.stack([dL_var_s / (rms[0] if rms[0] > 0 else 1.0),
                            dL_core_s / (rms[1] if rms[1] > 0 else 1.0),
                            dR_s / (rms[2] if rms[2] > 0 else 1.0)], axis=1)
        onehot = np.zeros((classes.size, len(class_ids)), dtype=np.float64)
        index = {str(c): i for i, c in enumerate(class_ids)}
        for i, c in enumerate(classes.tolist()):
            j = index.get(str(c))
            if j is not None:
                onehot[i, j] = 1.0
        if onehot.shape[1] + shifted.shape[1] != beta.size:
            raise RuntimeError("gauge design width does not match the frozen coefficients")
        pred_shift = np.concatenate([onehot, shifted], axis=1) @ beta
        err_shift = y - pred_shift
        orig = np.asarray(unit["residuals"]["TRACE"], dtype=np.float64)
        # the same/different-n_heavy subset split
        same_mask = np.asarray(same)
        out["units"][key] = {
            "n_pairs": int(len(y)),
            "identical_gauge_invariant_d_pred": float(np.max(np.abs(d_pred_check - np.asarray([p["d_pred"] for p in meta_pairs])))),
            "audit_mae_trace_unshifted": component_equal_mae(y - orig, comps),
            "audit_mae_trace_shifted": component_equal_mae(err_shift, comps),
            "abs_change": float(abs(component_equal_mae(err_shift, comps) - component_equal_mae(y - orig, comps))),
            "n_same_var_heavy_pairs": int(same_mask.sum()),
            "n_diff_var_heavy_pairs": int((~same_mask).sum()),
            "audit_mae_trace_shifted_same_var_heavy": component_equal_mae(err_shift[same_mask], comps[same_mask])
            if same_mask.any()
            else None,
            "audit_mae_trace_shifted_diff_var_heavy": component_equal_mae(err_shift[~same_mask], comps[~same_mask])
            if (~same_mask).any()
            else None,
            "audit_mae_trace_unshifted_same_var_heavy": component_equal_mae((y - orig)[same_mask], comps[same_mask])
            if same_mask.any()
            else None,
            "audit_mae_trace_unshifted_diff_var_heavy": component_equal_mae((y - orig)[~same_mask], comps[~same_mask])
            if (~same_mask).any()
            else None,
        }
    return out


def _audit_ds(unit: Mapping[str, Any], traces) -> dict[str, Any]:
    rows = unit["audit_pairs"]
    X = []
    for p in rows:
        tr = traces.get((unit["model"], int(unit["seed"]), p["pair_id"]))
        X.append([float(tr["dL_var_raw"]), float(tr["dL_core_raw"]), float(tr["dR_raw"])])
    return {
        "rows": rows,
        "components": np.asarray([p["component"] for p in rows], dtype=object),
        "X_cont": {"TRACE": np.asarray(X, dtype=np.float64)},
        "y": np.asarray([p["e_delta"] for p in rows], dtype=np.float64),
        "classes": np.asarray([p["class_id"] for p in rows], dtype=object),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["fit", "audit", "report"])
    ap.add_argument("--models", nargs="*", default=list(C.MODELS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(C.SEEDS))
    ap.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_REPLICATES)
    args = ap.parse_args(argv)
    if args.stage == "fit":
        stage_fit(args.models, args.seeds)
    elif args.stage == "audit":
        stage_audit()
    else:
        stage_report(int(args.n_boot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
