"""CLI orchestrator for the edit-context screen (eligibility, then scoring)."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    os.environ.setdefault("OMP_NUM_THREADS", "4")


_configure_imports()

import ec_common as C  # noqa: E402
import ec_score as S  # noqa: E402
import ec_units as U  # noqa: E402


def stage_eligibility() -> dict[str, Any]:
    pairs = U.load_pairs()
    payload = U.eligibility(pairs)
    C.write_json(C.ANALYSIS / "eligibility.json", payload)
    structured = U.attach_structure(pairs)
    df = U.eligible_pairs(structured)
    units = U.build_units(df)
    U.units_frame(units).to_csv(C.ANALYSIS / "context_units.csv", index=False)
    C.log(
        f"[eligibility] status={payload['status']} families={payload['counts']['n_families_with_comparisons']} "
        f"components={payload['counts']['n_components_in_comparisons']} "
        f"comparisons={payload['counts']['n_comparisons']} "
        f"greedy={payload['counts']['n_greedy_nonoverlapping_comparisons']}"
    )
    return payload


def _family_gains_mean(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    parts = [S.family_gains(df, c).set_index("class_id") for c in cols]
    n_cmp = parts[0]["n_comparisons"]
    frame = pd.DataFrame(
        {
            "n_comparisons": n_cmp,
            "mae_zero": np.mean([p["mae_zero"] for p in parts], axis=0),
            "mae_pred": np.mean([p["mae_pred"] for p in parts], axis=0),
            "gain": np.mean([p["gain"] for p in parts], axis=0),
        }
    )
    return frame.reset_index()


def stage_score() -> dict[str, Any]:
    elig = C.read_json(C.ANALYSIS / "eligibility.json")
    if elig["status"] != "OK":
        C.log(f"[score] blocked: eligibility status {elig['status']}")
        return {"status": elig["status"], "scored": False}

    pairs = U.load_pairs()
    structured = U.attach_structure(pairs)
    df_pairs = U.eligible_pairs(structured)
    units = U.build_units(df_pairs)
    comparisons = U.build_comparisons(units)
    greedy = U.greedy_non_overlapping(comparisons)

    labels = S.load_train_labels()
    traces = S.load_traces()
    integrity = {
        "labels": {
            "source": labels["source"],
            "label_scope": labels["label_scope"],
            "n_train_rows": int(labels["n_train_rows"]),
            "split_rows_sha256": labels["split_rows_sha256"],
            "csv_sha256": labels["csv_sha256"],
            "test_targets_decoded": False,
        },
        "traces": S.verify_traces(traces, pairs),
        "audit_dy_crosscheck": S.verify_audit_dy(labels, traces, pairs),
        "pair_level_audit": S.pair_level_audit_check(labels, traces, structured),
    }
    if not integrity["traces"]["ok"]:
        raise RuntimeError(f"trace verification failed: {integrity['traces']}")
    if not integrity["audit_dy_crosscheck"]["ok"]:
        raise RuntimeError(f"Audit dy cross-check failed: {integrity['audit_dy_crosscheck']}")
    if not integrity["pair_level_audit"]["ok"]:
        raise RuntimeError(f"pair-level audit reproduction failed: {integrity['pair_level_audit']}")

    units_df = S.unit_effects(units, labels, traces)
    comp_df = S.comparison_table(units_df, comparisons)
    if len(comp_df) != int(elig["counts"]["n_comparisons"]):
        raise RuntimeError("comparison table size changed between eligibility and scoring")
    units_df.to_csv(C.ANALYSIS / "context_units.csv", index=False)
    comp_df.to_csv(C.ANALYSIS / "context_comparisons.csv", index=False)

    pred_cols = [f"context_pred_{u}" for u in C.MODEL_UNITS]
    models: dict[str, Any] = {}
    family_tables: dict[str, pd.DataFrame] = {}
    for unit_name, col in zip(C.MODEL_UNITS, pred_cols):
        gain = S.primary_gain(comp_df, col)
        fam = S.family_gains(comp_df, col)
        family_tables[unit_name] = fam
        models[unit_name] = {
            "family_equal_gain": gain["family_equal_gain"],
            "family_equal_mae_zero": gain["family_equal_mae_zero"],
            "family_equal_mae_pred": gain["family_equal_mae_pred"],
            "pooled_gain": gain["pooled_gain"],
            "n_families": gain["n_families"],
            "n_comparisons": gain["n_comparisons"],
            "bootstrap": S.bootstrap_families(fam, replicates=C.BOOTSTRAP_REPLICATES, seed=C.RANDOM_SEED),
            "leave_one_family_out": S.leave_one_family_out(fam),
            "direction_accuracy": S.direction_accuracy(comp_df, col),
        }

    main_cols = [f"context_pred_{C.MAIN_MODEL}:s{s}" for s in C.SEEDS]
    ref_cols = [f"context_pred_{C.REFERENCE_MODEL}:s{s}" for s in C.SEEDS]
    main_fam = _family_gains_mean(comp_df, main_cols)
    ref_fam = _family_gains_mean(comp_df, ref_cols)
    main_fam.to_csv(C.ANALYSIS / "context_family_gains.csv", index=False)
    seed_mean = {
        "main_model": {
            "model": C.MAIN_MODEL,
            "columns": main_cols,
            "mean_of_seed_gains": float(np.mean([models[f"{C.MAIN_MODEL}:s{s}"]["family_equal_gain"] for s in C.SEEDS])),
            "family_equal_gain_of_seed_mean": float(main_fam["gain"].mean()),
            "bootstrap": S.bootstrap_families(main_fam, replicates=C.BOOTSTRAP_REPLICATES, seed=C.RANDOM_SEED + 1),
            "leave_one_family_out": S.leave_one_family_out(main_fam),
        },
        "reference_model": {
            "model": C.REFERENCE_MODEL,
            "columns": ref_cols,
            "mean_of_seed_gains": float(np.mean([models[f"{C.REFERENCE_MODEL}:s{s}"]["family_equal_gain"] for s in C.SEEDS])),
            "family_equal_gain_of_seed_mean": float(ref_fam["gain"].mean()),
            "bootstrap": S.bootstrap_families(ref_fam, replicates=C.BOOTSTRAP_REPLICATES, seed=C.RANDOM_SEED + 2),
            "leave_one_family_out": S.leave_one_family_out(ref_fam),
        },
    }

    perm = S.permutation_control(
        comp_df,
        units_df,
        {u: f"dp_{u}" for u in C.MODEL_UNITS},
        replicates=C.PERMUTATION_REPLICATES,
        seed=C.RANDOM_SEED,
    )
    C.write_json(C.ANALYSIS / "permutation_control.json", perm)
    perm_seed_mean = perm["main_seed_mean_null"]
    for unit_name in C.MODEL_UNITS:
        observed = float(models[unit_name]["family_equal_gain"])
        null = perm["units"][unit_name]
        models[unit_name]["permutation"] = {
            "null_median": null["null_median"],
            "null_q95": null["null_q95"],
            "observed": observed,
            "beats_null_median": bool(observed > null["null_median"]),
            "beats_null_q95": bool(observed > null["null_q95"]),
        }

    main_family_table = main_fam.sort_values("gain", ascending=False).reset_index(drop=True)
    total_gain = float(main_family_table["gain"].sum())
    influence = {
        "family_equal_mean_gain": float(main_family_table["gain"].mean()),
        "median_family_gain": float(main_family_table["gain"].median()),
        "n_families_positive": int((main_family_table["gain"] > 0).sum()),
        "n_families_negative": int((main_family_table["gain"] < 0).sum()),
        "n_families": int(len(main_family_table)),
        "top_family": {
            "class_id": str(main_family_table.iloc[0]["class_id"]),
            "gain": float(main_family_table.iloc[0]["gain"]),
            "n_comparisons": int(main_family_table.iloc[0]["n_comparisons"]),
            "share_of_positive_gain": float(
                main_family_table.iloc[0]["gain"] / total_gain if total_gain else float("nan")
            ),
            "share_of_mean": float(main_family_table.iloc[0]["gain"] / len(main_family_table)),
        },
        "top3_share_of_total_gain": float(
            main_family_table["gain"].head(3).sum() / total_gain if total_gain else float("nan")
        ),
        "per_family": main_family_table.to_dict(orient="records"),
        "per_seed_family_equal_gain": {
            f"{C.MAIN_MODEL}:s{s}": float(family_tables[f"{C.MAIN_MODEL}:s{s}"]["gain"].mean()) for s in C.SEEDS
        },
    }

    decision = S.decide(
        {u: models[u]["family_equal_gain"] for u in C.MODEL_UNITS},
        seed_mean_null_median=float(perm_seed_mean["median"]),
        seed_mean_loo=seed_mean["main_model"]["leave_one_family_out"],
        seed_mean_bootstrap=seed_mean["main_model"]["bootstrap"],
        support_ok=True,
        support_counts=elig["gates"],
    )

    payload = {
        "schema": "lcrc-edit-context-scores-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "environment": C.environment(),
        "inputs": C.input_manifest(),
        "exposure": {
            "historical_audit_exposed": True,
            "audit_labels_were_read_once_in_the_historical_round": True,
            "independent_confirmation": False,
            "note": "every number here is EXPLORATORY_EXPOSED_DATA; it can support a resource decision, not a claim of chemical truth",
        },
        "metric_definition": {
            "dy_p": "y(B_p) - y(A_p) in raw logD",
            "dp_p": "pred(B_p) - pred(A_p) in raw logD",
            "context_y": "dy_p - dy_q for two same-class units with different component and different core class",
            "context_pred": "dp_p - dp_q",
            "reference_prediction": "context_pred_zero = 0 (the same edit acts identically in every core)",
            "context_gain": "MAE(context_pred_zero, context_y) - MAE(context_pred, context_y)",
            "positive_means": "the frozen model beats the context-independent zero reference",
            "aggregation": "two MAEs inside each family, difference per family, then equal-weighted mean over families",
            "direction_accuracy": f"sign agreement on |context_y| >= {C.CONTEXT_EFFECT_THRESHOLD} raw logD (fixed descriptive stratum)",
        },
        "support": {
            "status": elig["status"],
            "counts": elig["counts"],
            "gates": elig["gates"],
            "greedy_non_overlapping": {
                "n_kept": greedy["n_kept"],
                "n_components_used": greedy["n_components_used"],
                "rule": greedy["rule"],
            },
        },
        "integrity": integrity,
        "units_table": {
            "n_units": int(len(units_df)),
            "n_units_single_pair": int((units_df["n_pairs_in_unit"] == 1).sum()),
            "n_units_multi_pair": int((units_df["n_pairs_in_unit"] > 1).sum()),
            "partition_counts": {k: int(v) for k, v in df_pairs["partition"].value_counts().sort_index().items()},
        },
        "comparisons": {
            "n_comparisons": int(len(comp_df)),
            "n_families": int(comp_df["class_id"].nunique()),
            "n_components": int(
                len(set(comp_df["component_p"]) | set(comp_df["component_q"]))
            ),
            "n_cores": int(len(set(comp_df["core_class_p"]) | set(comp_df["core_class_q"]))),
            "n_effect_subset": int(comp_df["in_effect_subset"].sum()),
            "mean_abs_context_y": float(comp_df["abs_context_y"].mean()),
            "median_abs_context_y": float(comp_df["abs_context_y"].median()),
            "max_abs_context_y": float(comp_df["abs_context_y"].max()),
        },
        "models": models,
        "seed_mean": seed_mean,
        "influence": influence,
        "permutation_control": {
            "file": C.track_rel(C.ANALYSIS / "permutation_control.json"),
            "replicates": C.PERMUTATION_REPLICATES,
            "seed": C.RANDOM_SEED,
            "seed_mean_null": {
                "median": perm_seed_mean["median"],
                "mean": perm_seed_mean["mean"],
                "q2.5": perm_seed_mean["q2.5"],
                "q97.5": perm_seed_mean["q97.5"],
                "q95": perm_seed_mean["q95"],
            },
            "design": perm["design"],
        },
        "decision": decision,
        "status": decision["status"],
    }
    C.write_json(C.ANALYSIS / "context_scores.json", payload)
    C.log(
        f"[score] C1 gains {['%.4f' % g for g in decision['main_seed_gains']]} "
        f"mean {decision['main_seed_mean_gain']:+.4f}; A0 mean {decision['reference_seed_mean_gain']:+.4f}; "
        f"status={decision['status']}"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("eligibility", "score", "all"))
    args = parser.parse_args(argv)
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage in ("eligibility", "all"):
            payload = stage_eligibility()
            if payload["status"] != "OK" and args.stage == "all":
                C.log("[eligibility] support gate failed; refusing to score")
                return 0
        if args.stage in ("score", "all"):
            out = stage_score()
            if out.get("status") == "INSUFFICIENT_CONTEXT_SUPPORT":
                C.log("[score] skipped: insufficient structural support")
        return 0
    except Exception as exc:  # noqa: BLE001 - recorded
        C.write_json(
            C.ANALYSIS / f"stage_{args.stage}_failure.json",
            {
                "stage": args.stage,
                "error": type(exc).__name__,
                "detail": str(exc),
                "traceback": traceback.format_exc(),
                "utc": C.utcnow(),
                "environment": C.environment(),
            },
        )
        C.log(f"STAGE FAILED {args.stage}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
