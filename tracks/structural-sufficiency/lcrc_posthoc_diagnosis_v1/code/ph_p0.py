"""P0: errata and the seed-conditional mean-difference bootstrap."""

from __future__ import annotations

import csv
import json
from typing import Any

import numpy as np

import ph_common as C

STRUCTURAL_MARGIN = 0.05


def _read_csv(path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def p0_1_counts() -> dict[str, Any]:
    mol_rows = _read_csv(C.PTB / "analysis/molecular_results.csv")
    tu_rows = _read_csv(C.PTB / "analysis/tu_results.csv")

    per_dataset: list[dict[str, Any]] = []
    for dataset in C.MOLECULE_ORDER:
        meta = C.DATASETS[dataset]
        vals = {}
        for model in C.MODELS:
            arr = [
                float(r["test_primary"])
                for r in mol_rows
                if r["dataset"] == dataset and r["model"] == model and r.get("status") == "COMPLETE"
            ]
            vals[model] = float(np.mean(arr))
        delta = vals["C1"] - vals["A0"]
        better = delta < 0 if meta["direction"] == "lower" else delta > 0
        per_dataset.append(
            {
                "dataset": dataset,
                "kind": "molecule",
                "metric": meta["primary"],
                "direction": meta["direction"],
                "A0_mean": vals["A0"],
                "C1_mean": vals["C1"],
                "delta_C1_minus_A0": delta,
                "C1_better": bool(better),
            }
        )

    for dataset in C.TU_ORDER:
        vals = {}
        for model in C.MODELS:
            arr = [
                float(r["test_accuracy"])
                for r in tu_rows
                if r["dataset"] == dataset and r["model"] == model and r.get("status") == "COMPLETE"
            ]
            vals[model] = float(np.mean(arr))
        delta = vals["C1"] - vals["A0"]
        per_dataset.append(
            {
                "dataset": dataset,
                "kind": "tu",
                "metric": "accuracy",
                "direction": "higher",
                "A0_mean": vals["A0"],
                "C1_mean": vals["C1"],
                "delta_C1_minus_A0": delta,
                "C1_better": bool(delta > 0),
            }
        )

    n_better = sum(1 for r in per_dataset if r["C1_better"])
    n_worse = len(per_dataset) - n_better
    n_better_mol = sum(1 for r in per_dataset if r["kind"] == "molecule" and r["C1_better"])
    n_better_tu = sum(1 for r in per_dataset if r["kind"] == "tu" and r["C1_better"])
    return {
        "n_datasets": len(per_dataset),
        "n_C1_better": n_better,
        "n_C1_worse": n_worse,
        "n_C1_better_molecule": n_better_mol,
        "n_C1_better_tu": n_better_tu,
        "original_text": "C1 在 4/10 更好、6/10 更差",
        "instruction_claimed_text": "C1 在 5/10 更好、5/10 更差",
        "corrected_text": f"C1 在 {n_better}/10 更好、{n_worse}/10 更差",
        "arithmetic_note": (
            "The task instruction's own per-task list (5 molecule tasks better + "
            "IMDB-MULTI better) sums to 6 better, not 5.  Recomputing all 10 dataset-level "
            "means gives 6/10 better and 4/10 worse; the old 4/10-better text is wrong, and "
            "the instruction's 5/5 total is also arithmetically inconsistent with its list."
        ),
        "per_dataset": per_dataset,
    }


def p0_2_counts() -> dict[str, Any]:
    audit = C.read_json(C.PTB / "analysis/mechanism_audit.json")["intervention"]
    rows = audit["rows"]
    out_rows = []
    for r in rows:
        iv = r["interventions"]
        means = {
            "typed_s2": float(iv["typed_s2_off"]["mean_abs_delta"]),
            "global_g": float(iv["global_g_off"]["mean_abs_delta"]),
            "rho_mean_h2": float(iv["rho_mean_h2_off"]["mean_abs_delta"]),
        }
        ordered = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
        best, second = ordered[0], ordered[1]
        margin = (best[1] - second[1]) / second[1] if second[1] else float("inf")
        out_rows.append(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "seed": int(r["seed"]),
                "scope": r["scope"],
                "means": means,
                "strict_best": best[0],
                "strict_typed_s2_best": best[0] == "typed_s2",
                f"{STRUCTURAL_MARGIN:.0%}_margin_best": best[0] if margin >= STRUCTURAL_MARGIN else "comparable",
                f"{STRUCTURAL_MARGIN:.0%}_margin_typed_s2_best": bool(best[0] == "typed_s2" and margin >= STRUCTURAL_MARGIN),
                "margin_over_second": float(margin),
            }
        )
    new_rows = [r for r in out_rows if r["scope"] == "THIS_ROUND_FROZEN_CHECKPOINT"]
    all_rows = out_rows
    strict_new = sum(r["strict_typed_s2_best"] for r in new_rows)
    strict_all = sum(r["strict_typed_s2_best"] for r in all_rows)
    margin_key = f"{STRUCTURAL_MARGIN:.0%}_margin_typed_s2_best"
    margin_new = sum(r[margin_key] for r in new_rows)
    margin_all = sum(r[margin_key] for r in all_rows)
    return {
        "n_rows": len(all_rows),
        "n_new_checkpoints": len(new_rows),
        "n_with_lipophilicity": len(all_rows),
        "scope_counts": {
            "new_12_strict_max": f"{strict_new}/{len(new_rows)}",
            "new_12_5pct_margin": f"{margin_new}/{len(new_rows)}",
            "all_13_strict_max": f"{strict_all}/{len(all_rows)}",
            "all_13_5pct_margin": f"{margin_all}/{len(all_rows)}",
        },
        "reported": {"summary": "10/12", "body": "11/13"},
        "interpretation": (
            "The summary 10/12 and body 11/13 refer to the same convention: 12 new "
            "checkpoints plus the 1 historical Lipophilicity checkpoint = 13, with the "
            "two MolHIV rows treated as 'comparable' exceptions.  The count is a scope "
            "difference, not a contradiction.  Under a strict numeric maximum the count "
            "is 11/12 new and 12/13 all; MolHIV:C1 typed-S2 is numerically highest but only "
            "3.0% above mean_h2, hence 'comparable' under the 5% margin rule.  "
            "This is a fixed-weight zeroing sensitivity comparison of input blocks of "
            "different widths, not a trained S2-necessity result."
        ),
        "rows": out_rows,
    }


def _load_dataset(dataset: str):
    meta = C.DATASETS[dataset]
    z = {}
    for seed in C.MOLECULE_SEEDS:
        a0 = C.load_mol(dataset, "A0", seed)
        c1 = C.load_mol(dataset, "C1", seed)
        rows = np.asarray(a0["rows"], dtype=np.int64)
        if not np.array_equal(rows, np.asarray(c1["rows"], dtype=np.int64)):
            raise RuntimeError(f"{dataset}/s{seed}: rows differ")
        y = np.asarray(a0["y_true"], dtype=np.float64)
        if not np.allclose(y, np.asarray(c1["y_true"], dtype=np.float64), equal_nan=True):
            raise RuntimeError(f"{dataset}/s{seed}: y differs")
        z[seed] = {
            "rows": rows,
            "y": y,
            "A0": np.asarray(a0["pred"], dtype=np.float64),
            "C1": np.asarray(c1["pred"], dtype=np.float64),
        }
    base = z[C.MOLECULE_SEEDS[0]]
    for seed in C.MOLECULE_SEEDS[1:]:
        if not np.array_equal(z[seed]["rows"], base["rows"]):
            raise RuntimeError(f"{dataset}: rows differ across seeds")
        if not np.allclose(z[seed]["y"], base["y"], equal_nan=True):
            raise RuntimeError(f"{dataset}: y differs across seeds")
    mask = C.original_task_mask(base["y"]) if meta["task"] == "multitask" else None
    return z, mask


def mean_seed_bootstrap() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    old = C.read_json(C.PTB / "analysis/statistics.json")["molecule_paired"]
    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for dataset in C.MOLECULE_ORDER:
        meta = C.DATASETS[dataset]
        z, mask = _load_dataset(dataset)
        n = z[0]["rows"].size
        observed = {}
        for seed in C.MOLECULE_SEEDS:
            a0 = C.primary_metric(dataset, z[seed]["A0"], z[seed]["y"], mask)
            c1 = C.primary_metric(dataset, z[seed]["C1"], z[seed]["y"], mask)
            observed[seed] = {"A0": a0, "C1": c1, "delta": c1 - a0}
        delta_seed = np.asarray([observed[s]["delta"] for s in C.MOLECULE_SEEDS], dtype=np.float64)
        delta_mean = float(delta_seed.mean())

        rng = np.random.default_rng(C.BOOTSTRAP_SEED)
        reps: list[float] = []
        attempts = 0
        invalid = 0
        while len(reps) < C.BOOTSTRAP_REPLICATES and attempts < C.BOOTSTRAP_MAX_ATTEMPTS:
            attempts += 1
            if meta["task"] == "regression":
                idx = C.bootstrap_rows(n, rng)
            elif meta["task"] == "binary":
                idx = C.bootstrap_stratified(z[0]["y"], rng)
            else:
                idx = C.bootstrap_rows(n, rng)
                yy = z[0]["y"][idx]
                ok = True
                for j in mask:
                    obs = np.isfinite(yy[:, j])
                    if obs.sum() == 0 or np.unique(yy[obs, j]).size != 2:
                        ok = False
                        break
                if not ok:
                    invalid += 1
                    continue
            per_seed = []
            for seed in C.MOLECULE_SEEDS:
                a0 = C.primary_metric(dataset, z[seed]["A0"][idx], z[seed]["y"][idx], mask)
                c1 = C.primary_metric(dataset, z[seed]["C1"][idx], z[seed]["y"][idx], mask)
                per_seed.append(c1 - a0)
            reps.append(float(np.mean(per_seed)))

        status = "OK" if len(reps) >= C.BOOTSTRAP_REPLICATES else "UNAVAILABLE"
        ci = C.quantile_ci(reps) if status == "OK" else {"lo": None, "hi": None, "n_finite": len(reps)}
        old_seed = old[dataset]["per_seed"]
        envelope = {
            "lo": float(min(p["boot_lo"] for p in old_seed)),
            "hi": float(max(p["boot_hi"] for p in old_seed)),
            "label": "SEED_WISE_CI_ENVELOPE",
        }
        spans_zero = bool(status == "OK" and (ci["lo"] <= 0 <= ci["hi"]))
        frac_positive = float(np.mean(np.asarray(reps) > 0)) if reps else None
        rec = {
            "dataset": dataset,
            "task": meta["task"],
            "primary_metric": meta["primary"],
            "direction": meta["direction"],
            "n_test_rows": int(n),
            "n_seeds": len(C.MOLECULE_SEEDS),
            "per_seed": {str(s): observed[s] for s in C.MOLECULE_SEEDS},
            "seed_deltas": [float(x) for x in delta_seed],
            "delta_mean": delta_mean,
            "delta_seed_sd": float(delta_seed.std(ddof=1)),
            "bootstrap": {
                "replicates_requested": C.BOOTSTRAP_REPLICATES,
                "replicates_valid": len(reps),
                "attempts": attempts,
                "invalid_draws": invalid,
                "seed": C.BOOTSTRAP_SEED,
                "status": status,
                "ci_lo": ci["lo"],
                "ci_hi": ci["hi"],
                "frac_positive": frac_positive,
                "spans_zero": spans_zero,
                "definition": (
                    "seed-conditional mean-difference bootstrap: same row resample for all "
                    "seeds and both models; 2.5/97.5 percentiles"
                ),
            },
            "old_seed_wise_envelope": envelope,
            "old_summary_ci_spans_zero": bool(old[dataset]["ci_spans_zero"]),
            "corrected_ci_spans_zero": spans_zero,
            "label": C.RESULT_LABEL,
        }
        if meta["task"] == "multitask":
            rec["tox21_tasks_used"] = [int(j) for j in mask]
            rec["n_tasks_used"] = len(mask)
        records.append(rec)
        csv_rows.append(
            {
                "dataset": dataset,
                "task": meta["task"],
                "metric": meta["primary"],
                "A0_s0": observed[0]["A0"],
                "A0_s1": observed[1]["A0"],
                "A0_s2": observed[2]["A0"],
                "C1_s0": observed[0]["C1"],
                "C1_s1": observed[1]["C1"],
                "C1_s2": observed[2]["C1"],
                "delta_s0": observed[0]["delta"],
                "delta_s1": observed[1]["delta"],
                "delta_s2": observed[2]["delta"],
                "delta_mean": delta_mean,
                "delta_seed_sd": float(delta_seed.std(ddof=1)),
                "mean_seed_ci_lo": ci["lo"],
                "mean_seed_ci_hi": ci["hi"],
                "mean_seed_ci_status": status,
                "mean_seed_frac_positive": frac_positive,
                "old_envelope_lo": envelope["lo"],
                "old_envelope_hi": envelope["hi"],
                "old_envelope_label": "SEED_WISE_CI_ENVELOPE",
                "n_test_rows": int(n),
                "result_label": C.RESULT_LABEL,
            }
        )

    payload = {
        "schema": "lcrc-posthoc-mean-seed-bootstrap-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "definition": (
            "For each dataset, resample the same Test rows for all three seeds and both "
            "models; recompute each seed's primary-metric difference; average the three "
            "differences.  2000 replicates, seed 20261103, percentile 2.5/97.5.  "
            "Conditional on the trained checkpoints; seed variation reported separately."
        ),
        "old_envelope_definition": (
            "min(per-seed boot_lo), max(per-seed boot_hi) from the previous round; not a CI "
            "for the mean across seeds; do not use it as one."
        ),
        "datasets": records,
        "n_datasets_corrected_ci_spans_zero": sum(r["corrected_ci_spans_zero"] for r in records),
        "n_datasets_old_envelope_spans_zero": sum(r["old_summary_ci_spans_zero"] for r in records),
    }
    return payload, csv_rows


def main() -> int:
    errata = {
        "schema": "lcrc-posthoc-p0-errata-v1",
        "round_id": C.ROUND_ID,
        "baseline_commit": C.BASELINE_COMMIT,
        "result_label": C.RESULT_LABEL,
        "P0_1_full_c1_count": p0_1_counts(),
        "P0_2_typed_s2_scope": p0_2_counts(),
        "P0_3_ci_definition": {
            "old_label": "SEED_WISE_CI_ENVELOPE",
            "old_code": "code/ptb_report.py::molecule_summary: min(boot_lo), max(boot_hi)",
            "old_is_not": "95% CI for the mean C1-A0 effect across seeds",
            "new_label": "SEED_CONDITIONAL_MEAN_DIFFERENCE_BOOTSTRAP_CI",
            "historical_statistics_json_modified": False,
        },
    }
    C.write_json(C.ANALYSIS / "p0_errata.json", errata)
    payload, rows = mean_seed_bootstrap()
    C.write_json(C.ANALYSIS / "mean_seed_bootstrap_ci.json", payload)
    C.save_csv(
        C.ANALYSIS / "mean_seed_bootstrap_ci.csv",
        rows,
        columns=list(rows[0].keys()),
    )
    print(json.dumps(errata["P0_1_full_c1_count"], indent=1)[:600])
    print(json.dumps(errata["P0_2_typed_s2_scope"]["scope_counts"], indent=1))
    for r in payload["datasets"]:
        b = r["bootstrap"]
        print(
            r["dataset"],
            f"delta_mean={r['delta_mean']:+.5f}",
            f"new_CI=[{b['ci_lo']:+.5f},{b['ci_hi']:+.5f}]",
            f"old_env=[{r['old_seed_wise_envelope']['lo']:+.5f},{r['old_seed_wise_envelope']['hi']:+.5f}]",
            f"spans0={b['spans_zero']}",
            f"valid={b['replicates_valid']}/{b['attempts']}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
