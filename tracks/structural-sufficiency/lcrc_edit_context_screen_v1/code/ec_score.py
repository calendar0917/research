"""Context-effect scoring on the exposed historical Diagnose/Audit pairs.

Definitions (fixed by the execution instruction):

    dy_p          = y(B_p) - y(A_p)                      raw logD
    dp_p          = pred(B_p) - pred(A_p)                raw logD
    context_y     = dy_p - dy_q                          same class_id, other core
    context_pred  = dp_p - dp_q
    ContextGain   = MAE(0, context_y) - MAE(context_pred, context_y)

Aggregation is family-equal: compute the two MAEs inside every transform
family, take the difference, then average over families.  No model is trained
here; the frozen per-pair predictions come from the historical trace file.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import ec_common as C

for _p in (str(C.HIST_CODE), str(C.DISSECT_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# frozen label source
# ---------------------------------------------------------------------------


def load_train_labels() -> dict[str, Any]:
    """Train-part targets through the historical frozen repeat-122 loader."""
    import dissect_source as S

    src = S.load_lipop(C.LABEL_REPEAT, labels=C.LABEL_SCOPE)
    train_rows = [int(r) for r in src["rows"]["train"]]
    y = np.asarray(src["y"]["train"], dtype=np.float64)
    if len(train_rows) != len(y):
        raise ValueError("label loader returned mismatched rows/targets")
    return {
        "row_to_y": {r: float(v) for r, v in zip(train_rows, y)},
        "n_train_rows": len(train_rows),
        "label_scope": C.LABEL_SCOPE,
        "source": C.LABEL_SOURCE,
        "split_rows_sha256": list(src["split_rows_sha256"]),
        "csv_sha256": src["csv_sha256"],
    }


def pair_dy(labels: Mapping[str, Any], row_a: int, row_b: int) -> float:
    r2y = labels["row_to_y"]
    a, b = int(row_a), int(row_b)
    if a not in r2y or b not in r2y:
        raise KeyError(f"pair rows {(a, b)} are outside the published Train part")
    return float(r2y[b] - r2y[a])


# ---------------------------------------------------------------------------
# frozen per-pair traces
# ---------------------------------------------------------------------------


def load_traces() -> pd.DataFrame:
    df = pd.read_csv(C.TRACES_CSV, usecols=list(C.TRACE_COLUMNS))
    df["unit"] = df["model"].astype(str) + ":s" + df["seed"].astype(int).astype(str)
    df["pair_id"] = df["pair_id"].astype(str)
    if df.duplicated(["pair_id", "unit"]).any():
        raise ValueError("pair_traces.csv has duplicate (pair_id, unit) rows")
    unknown = sorted(set(df["unit"]) - set(C.MODEL_UNITS))
    if unknown:
        raise ValueError(f"unexpected model units in the traces: {unknown}")
    missing = sorted(set(C.MODEL_UNITS) - set(df["unit"]))
    if missing:
        raise ValueError(f"missing model units in the traces: {missing}")
    return df


def verify_traces(traces: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    per_unit = traces.groupby("unit")["pair_id"].nunique()
    n_pairs = int(pairs["pair_id"].nunique())
    row_max = float(
        np.max(np.abs(traces["pred_b_raw"].to_numpy() - traces["pred_a_raw"].to_numpy() - traces["d_pred_raw"].to_numpy()))
    )
    finite = bool(np.isfinite(traces[["pred_a_raw", "pred_b_raw", "d_pred_raw"]].to_numpy()).all())
    rows_match = bool(
        (traces["row_a"].to_numpy() == traces["pair_id"].map(dict(zip(pairs["pair_id"], pairs["row_a"]))).to_numpy()).all()
        and (
            traces["row_b"].to_numpy()
            == traces["pair_id"].map(dict(zip(pairs["pair_id"], pairs["row_b"]))).to_numpy()
        ).all()
    )
    return {
        "n_trace_rows": int(len(traces)),
        "n_pairs": n_pairs,
        "n_units": int(traces["unit"].nunique()),
        "pairs_per_unit": {k: int(v) for k, v in per_unit.items()},
        "all_units_cover_all_pairs": bool((per_unit == n_pairs).all()),
        "max_abs(d_pred - (pred_b - pred_a))": row_max,
        "d_pred_identity_ok": bool(row_max <= 1e-6),
        "all_predictions_finite": finite,
        "pair_rows_match": rows_match,
        "ok": bool((per_unit == n_pairs).all() and row_max <= 1e-6 and finite and rows_match),
    }


def verify_audit_dy(labels: Mapping[str, Any], traces: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    """Cross-check label-derived dy against the frozen Audit score file."""
    audit = C.read_json(C.AUDIT_SCORES)
    trace_lookup = {
        (str(r.pair_id), str(r.unit)): float(r.d_pred_raw)
        for r in traces.itertuples(index=False)
    }
    dy_lookup = {
        str(r.pair_id): pair_dy(labels, int(r.row_a), int(r.row_b)) for r in pairs.itertuples(index=False)
    }
    max_pred = 0.0
    max_dy = 0.0
    n = 0
    for unit, payload in audit["units"].items():
        for rec in payload["audit_pairs"]:
            pid = str(rec["pair_id"])
            max_pred = max(max_pred, abs(float(rec["d_pred"]) - trace_lookup[(pid, unit)]))
            implied_dy = float(rec["d_pred"]) - float(rec["e_delta"])
            max_dy = max(max_dy, abs(implied_dy - dy_lookup[pid]))
            n += 1
    return {
        "n_audit_records_checked": int(n),
        "n_audit_pairs": int(len(audit["units"]["C1:s0"]["audit_pairs"])),
        "max_abs_trace_minus_probe_d_pred": float(max_pred),
        "max_abs_label_dy_minus_probe_implied_dy": float(max_dy),
        "ok": bool(n > 0 and max_pred <= 1e-9 and max_dy <= 1e-9),
    }


def pair_level_audit_check(
    labels: Mapping[str, Any], traces: pd.DataFrame, pairs: pd.DataFrame
) -> dict[str, Any]:
    """Recompute the historical Audit NATIVE MAE from labels + traces.

    The historical ``probe_audit_scores.json`` stores, per model unit, the
    component-equal-weight MAE of ``|d_pred - d_y|`` on the 74 Audit pairs for
    its never-penalised zero predictor (``NATIVE``).  Reproducing those frozen
    numbers from our own label mapping is an end-to-end check that
    ``(d_pred, d_y, pair)`` are aligned.
    """
    audit = C.read_json(C.AUDIT_SCORES)
    structured = pairs.copy()
    comp = {s["pair_id"]: s["component"] for s in attach_component_records(structured)}
    audit_pairs = {str(r.pair_id) for r in structured.itertuples(index=False) if str(r.partition) == "audit"}
    dy = {
        str(r.pair_id): pair_dy(labels, int(r.row_a), int(r.row_b))
        for r in structured.itertuples(index=False)
        if str(r.pair_id) in audit_pairs
    }
    out: dict[str, Any] = {"units": {}, "n_audit_pairs": int(len(audit_pairs))}
    max_delta = 0.0
    for unit_name in C.MODEL_UNITS:
        frozen = float(audit["units"][unit_name]["scores"]["NATIVE"]["component_equal_weight_mae"])
        sub = traces[(traces["unit"] == unit_name) & (traces["pair_id"].isin(audit_pairs))]
        if sub["pair_id"].nunique() != len(audit_pairs):
            raise RuntimeError(f"audit trace coverage incomplete for {unit_name}")
        resid = np.abs(sub["d_pred_raw"].to_numpy() - np.asarray([dy[p] for p in sub["pair_id"]], dtype=np.float64))
        frame = pd.DataFrame({"component": [comp[p] for p in sub["pair_id"]], "resid": resid})
        recomputed = float(frame.groupby("component")["resid"].mean().mean())
        delta = abs(recomputed - frozen)
        max_delta = max(max_delta, delta)
        out["units"][unit_name] = {
            "frozen_component_equal_weight_mae": frozen,
            "recomputed_component_equal_weight_mae": recomputed,
            "abs_delta": delta,
        }
    out["max_abs_delta"] = float(max_delta)
    out["ok"] = bool(max_delta <= 1e-9)
    return out


def attach_component_records(pairs: pd.DataFrame) -> list[dict[str, Any]]:
    """Pair -> component using the historical split (read-only)."""
    payload = C.read_json(C.COMPONENT_SPLIT)["row_to_component"]
    mapping = {int(k): str(v) for k, v in payload.items()}
    out = []
    for r in pairs.itertuples(index=False):
        out.append({"pair_id": str(r.pair_id), "component": mapping[int(r.row_a)]})
    return out


# ---------------------------------------------------------------------------
# unit and comparison tables
# ---------------------------------------------------------------------------


def unit_effects(
    units: Sequence[Mapping[str, Any]], labels: Mapping[str, Any], traces: pd.DataFrame
) -> pd.DataFrame:
    by_unit: dict[str, dict[str, float]] = {}
    for u in units:
        rows = [str(p) for p in u["pair_ids"]]
        by_unit[u["unit_id"]] = {}
        dy = []
        for pid in rows:
            rec = traces[traces["pair_id"] == pid].iloc[0]
            dy.append(pair_dy(labels, int(rec["row_a"]), int(rec["row_b"])))
        by_unit[u["unit_id"]]["dy"] = float(np.mean(dy))
        for unit_name in C.MODEL_UNITS:
            sub = traces[(traces["pair_id"].isin(rows)) & (traces["unit"] == unit_name)]
            if sub["pair_id"].nunique() != len(rows):
                raise RuntimeError(f"trace rows missing for {u['unit_id']} / {unit_name}")
            by_unit[u["unit_id"]][f"dp_{unit_name}"] = float(sub["d_pred_raw"].mean())
    df = pd.DataFrame(
        [
            {
                "unit_id": u["unit_id"],
                "class_id": u["class_id"],
                "core_class": int(u["core_class"]),
                "component": u["component"],
                "n_pairs_in_unit": int(u["n_pairs"]),
                "pair_ids": ";".join(u["pair_ids"]),
                "partitions": ";".join(u["partitions"]),
                **by_unit[u["unit_id"]],
            }
            for u in units
        ]
    )
    return df


def comparison_table(units_df: pd.DataFrame, comparisons: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    idx = {str(r["unit_id"]): r for r in units_df.to_dict(orient="records")}
    rows = []
    for c in comparisons:
        p = idx[str(c["unit_p"])]
        q = idx[str(c["unit_q"])]
        rec = {
            "comparison_id": c["comparison_id"],
            "class_id": c["class_id"],
            "unit_p": c["unit_p"],
            "unit_q": c["unit_q"],
            "component_p": c["component_p"],
            "component_q": c["component_q"],
            "core_class_p": int(c["core_class_p"]),
            "core_class_q": int(c["core_class_q"]),
            "pair_p": c["pair_p"],
            "pair_q": c["pair_q"],
            "dy_p": float(p["dy"]),
            "dy_q": float(q["dy"]),
            "context_y": float(p["dy"] - q["dy"]),
        }
        for unit_name in C.MODEL_UNITS:
            rec[f"context_pred_{unit_name}"] = float(p[f"dp_{unit_name}"] - q[f"dp_{unit_name}"])
        rec["abs_context_y"] = abs(rec["context_y"])
        rec["in_effect_subset"] = bool(rec["abs_context_y"] >= C.CONTEXT_EFFECT_THRESHOLD)
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("comparison_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# family-equal metric
# ---------------------------------------------------------------------------


def family_gains(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    tmp = pd.DataFrame(
        {
            "class_id": df["class_id"].to_numpy(),
            "e_zero": np.abs(df["context_y"].to_numpy()),
            "e_pred": np.abs(df[pred_col].to_numpy() - df["context_y"].to_numpy()),
        }
    )
    g = tmp.groupby("class_id", sort=True).agg(
        n_comparisons=("e_zero", "size"), mae_zero=("e_zero", "mean"), mae_pred=("e_pred", "mean")
    )
    g["gain"] = g["mae_zero"] - g["mae_pred"]
    return g.reset_index()


def primary_gain(df: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    g = family_gains(df, pred_col)
    return {
        "family_equal_gain": float(g["gain"].mean()),
        "family_equal_mae_zero": float(g["mae_zero"].mean()),
        "family_equal_mae_pred": float(g["mae_pred"].mean()),
        "pooled_gain": float(g["mae_zero"].sum() / g["n_comparisons"].sum() - g["mae_pred"].sum() / g["n_comparisons"].sum()),
        "n_families": int(g["n_comparisons"].size),
        "n_comparisons": int(g["n_comparisons"].sum()),
        "families": g.to_dict(orient="records"),
    }


def bootstrap_families(gains: pd.DataFrame, *, replicates: int, seed: int) -> dict[str, Any]:
    values = gains["gain"].to_numpy(dtype=np.float64)
    n = values.size
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(replicates), n))
    means = values[idx].mean(axis=1)
    return {
        "unit": "transform family",
        "replicates": int(replicates),
        "seed": int(seed),
        "n_families": int(n),
        "observed": float(values.mean()),
        "ci_2.5": float(np.percentile(means, 2.5)),
        "ci_97.5": float(np.percentile(means, 97.5)),
        "bootstrap_sd": float(means.std(ddof=1)),
        "note": "families may share components; the interval is an exploratory cluster estimate, not a significance test",
    }


def leave_one_family_out(gains: pd.DataFrame) -> dict[str, Any]:
    values = gains.set_index("class_id")["gain"]
    n = len(values)
    rows = []
    for fam in values.index:
        rest = values.drop(index=fam)
        rows.append(
            {
                "removed_family": str(fam),
                "n_families_left": int(len(rest)),
                "mean_gain_without": float(rest.mean()) if len(rest) else float("nan"),
                "delta_vs_full": float(rest.mean() - values.mean()) if len(rest) else float("nan"),
            }
        )
    df = pd.DataFrame(rows).sort_values("mean_gain_without")
    flips = df[df["mean_gain_without"] <= 0.0]["removed_family"].tolist()
    return {
        "n_families": int(n),
        "full_mean_gain": float(values.mean()),
        "min_mean_gain_without": float(df["mean_gain_without"].min()),
        "max_mean_gain_without": float(df["mean_gain_without"].max()),
        "max_abs_delta": float(df["delta_vs_full"].abs().max()),
        "families_whose_removal_flips_sign": flips,
        "any_flip": bool(flips),
        "table": df.to_dict(orient="records"),
    }


def direction_accuracy(df: pd.DataFrame, pred_col: str) -> dict[str, Any]:
    sub = df[df["in_effect_subset"]].copy()
    out: dict[str, Any] = {"threshold_abs_context_y": C.CONTEXT_EFFECT_THRESHOLD}
    if sub.empty:
        out.update({"n_comparisons": 0, "accuracy": None, "family_equal_accuracy": None, "ok": False})
        return out
    correct = (np.sign(sub[pred_col].to_numpy()) == np.sign(sub["context_y"].to_numpy())).astype(float)
    sub = sub.assign(_correct=correct)
    fam = sub.groupby("class_id", sort=True)["_correct"].agg(["size", "mean"])
    out.update(
        {
            "n_comparisons": int(len(sub)),
            "n_correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "n_families": int(len(fam)),
            "family_equal_accuracy": float(fam["mean"].mean()),
            "per_family": [
                {"class_id": str(k), "n_comparisons": int(v["size"]), "accuracy": float(v["mean"])}
                for k, v in fam.iterrows()
            ],
            "tie_policy": "a zero predicted context difference counts as incorrect",
            "ok": True,
        }
    )
    return out


# ---------------------------------------------------------------------------
# permutation control
# ---------------------------------------------------------------------------


def permutation_control(
    df: pd.DataFrame,
    units_df: pd.DataFrame,
    unit_cols: Mapping[str, str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Permute predicted ``dy`` across the independent cores of each family.

    ``unit_cols`` maps a reported unit name (``C1:s0``) to the unit-level
    predicted-dy column it must permute.
    """
    unit_ids = list(units_df["unit_id"].astype(str))
    pos = {u: i for i, u in enumerate(unit_ids)}
    fams: dict[str, list[int]] = defaultdict(list)
    for r in units_df.itertuples(index=False):
        fams[str(r.class_id)].append(pos[str(r.unit_id)])
    for k in fams:
        fams[k].sort()
    fam_names = sorted(fams)
    rng = np.random.default_rng(int(seed))
    # Shared permutations across models (common random numbers).
    perms = {k: [rng.permutation(len(fams[k])) for _ in range(int(replicates))] for k in fam_names}
    idx_p = np.asarray([pos[str(u)] for u in df["unit_p"]], dtype=np.int64)
    idx_q = np.asarray([pos[str(u)] for u in df["unit_q"]], dtype=np.int64)
    fam_of = df["class_id"].astype(str).to_numpy()
    context_y = df["context_y"].to_numpy(dtype=np.float64)
    out_units: dict[str, dict[str, Any]] = {}
    nulls: dict[str, np.ndarray] = {}
    for label, col in unit_cols.items():
        dp = units_df[col].to_numpy(dtype=np.float64)
        null = np.empty(int(replicates), dtype=np.float64)
        for r in range(int(replicates)):
            permuted = dp.copy()
            for k in fam_names:
                members = fams[k]
                perm = perms[k][r]
                permuted[members] = dp[[members[j] for j in perm]]
            ctx = permuted[idx_p] - permuted[idx_q]
            err = np.abs(ctx - context_y)
            zero = np.abs(context_y)
            frame = pd.DataFrame({"class_id": fam_of, "e0": zero, "e1": err})
            g = frame.groupby("class_id", sort=True)[["e0", "e1"]].mean()
            null[r] = float((g["e0"] - g["e1"]).mean())
        nulls[label] = null
        out_units[label] = {
            "null_median": float(np.median(null)),
            "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "null_q2.5": float(np.percentile(null, 2.5)),
            "null_q97.5": float(np.percentile(null, 97.5)),
            "null_q95": float(np.percentile(null, 95)),
            "replicates": int(replicates),
        }
    main_labels = [f"{C.MAIN_MODEL}:s{s}" for s in C.SEEDS if f"{C.MAIN_MODEL}:s{s}" in nulls]
    if not main_labels:
        raise RuntimeError("no main-model unit was permuted")
    null_seed_mean = np.mean(np.stack([nulls[c] for c in main_labels], axis=0), axis=0)
    return {
        "schema": "lcrc-edit-context-permutation-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "seed": int(seed),
        "replicates": int(replicates),
        "design": {
            "permuted_quantity": "predicted dy (dp) of each unit",
            "within": "transform family (class_id)",
            "constraint": "a random permutation of the family's unit-level predicted dy values across its independent cores",
            "preserved": "the family's predicted-dy distribution and the fixed true context_y values",
            "destroyed": "the one-to-one correspondence between structural context and predicted response",
            "shared_across_models": True,
        },
        "units": out_units,
        "main_seed_mean_null": {
            "units": main_labels,
            "median": float(np.median(null_seed_mean)),
            "mean": float(null_seed_mean.mean()),
            "sd": float(null_seed_mean.std(ddof=1)),
            "q2.5": float(np.percentile(null_seed_mean, 2.5)),
            "q97.5": float(np.percentile(null_seed_mean, 97.5)),
            "q95": float(np.percentile(null_seed_mean, 95)),
            "replicates": int(replicates),
        },
    }


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def decide(
    gains_by_unit: Mapping[str, float],
    *,
    seed_mean_null_median: float,
    seed_mean_loo: Mapping[str, Any],
    seed_mean_bootstrap: Mapping[str, Any],
    support_ok: bool,
    support_counts: Mapping[str, Any],
) -> dict[str, Any]:
    main_gains = [float(gains_by_unit[f"{C.MAIN_MODEL}:s{s}"]) for s in C.SEEDS]
    ref_gains = [float(gains_by_unit[f"{C.REFERENCE_MODEL}:s{s}"]) for s in C.SEEDS]
    mean_gain = float(np.mean(main_gains))
    criteria = {
        "at_least_two_positive_seeds": {
            "required": int(C.DECISION["min_positive_main_seeds"]),
            "observed": int(sum(1 for g in main_gains if g > 0.0)),
            "pass": int(sum(1 for g in main_gains if g > 0.0)) >= int(C.DECISION["min_positive_main_seeds"]),
        },
        "mean_gain_at_least_threshold": {
            "required": float(C.DECISION["min_mean_context_gain"]),
            "observed": mean_gain,
            "pass": mean_gain >= float(C.DECISION["min_mean_context_gain"]),
        },
        "beats_permutation_median": {
            "permutation_median": float(seed_mean_null_median),
            "observed": mean_gain,
            "pass": mean_gain > float(seed_mean_null_median),
        },
        "no_single_family_flip": {
            "families_that_flip": list(seed_mean_loo["families_whose_removal_flips_sign"]),
            "min_mean_gain_without": float(seed_mean_loo["min_mean_gain_without"]),
            "pass": not bool(seed_mean_loo["any_flip"]),
        },
    }
    reasons: list[str] = []
    if not support_ok:
        status = "INSUFFICIENT_CONTEXT_SUPPORT"
        reasons.append(
            "structural support below the preset gate: "
            + ", ".join(
                f"{k}={v['observed']}<{v['required']}" for k, v in support_counts.items() if not v["pass"]
            )
        )
    elif all(c["pass"] for c in criteria.values()):
        status = "PILOT_PROMISING"
        reasons.append("all four preset criteria passed")
    else:
        status = "PILOT_NOT_ESTABLISHED"
        failed = [k for k, v in criteria.items() if not v["pass"]]
        reasons.extend(f"failed: {k}" for k in failed)
        if not criteria["mean_gain_at_least_threshold"]["pass"]:
            reasons.append(
                f"effect too small: mean ContextGain {mean_gain:+.4f} < "
                f"{C.DECISION['min_mean_context_gain']:+.2f} raw logD"
            )
        if not criteria["no_single_family_flip"]["pass"]:
            reasons.append(
                "the direction depends on a single family: removing "
                f"{criteria['no_single_family_flip']['families_that_flip']} flips the mean gain"
            )
        if not criteria["beats_permutation_median"]["pass"]:
            reasons.append("the model does not beat its within-family permutation median")
        if not criteria["at_least_two_positive_seeds"]["pass"]:
            reasons.append("seed instability: fewer than two C1 seeds have a positive ContextGain")
    return {
        "status": status,
        "reasons": reasons,
        "criteria": criteria,
        "main_model": C.MAIN_MODEL,
        "reference_model": C.REFERENCE_MODEL,
        "main_seed_gains": main_gains,
        "main_seed_mean_gain": mean_gain,
        "reference_seed_gains": ref_gains,
        "reference_seed_mean_gain": float(np.mean(ref_gains)),
        "bootstrap": dict(seed_mean_bootstrap),
        "leave_one_family_out": dict(seed_mean_loo),
        "thresholds": dict(C.DECISION),
        "interpretation_boundary": "resource decision for buying an independent validation set, not a paper claim",
    }


__all__ = [
    "bootstrap_families",
    "comparison_table",
    "decide",
    "direction_accuracy",
    "family_gains",
    "leave_one_family_out",
    "load_train_labels",
    "load_traces",
    "pair_dy",
    "pair_level_audit_check",
    "permutation_control",
    "primary_gain",
    "unit_effects",
    "verify_audit_dy",
    "verify_traces",
]
