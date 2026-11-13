"""Stage ``weighted``: one additional, explicitly weighted response table.

Why this file exists
--------------------
The executed round stores two *different* response statistics in the same
``partB_*_metrics.json`` block:

* ``delta_mae`` / ``gain`` / ``baseline_zero_change`` come from
  ``analysis.molecule_means`` (molecule -> instance -> control -> donor, i.e.
  molecule-equal, then a plain mean over molecules);
* ``abs_delta_pred.{mean,median,p95,max}`` are computed from a **flattened**
  array of the T and K views of every selected pair (pair/view weighted).

Both are legitimate, but they answer different questions and the round's
REPORT quoted the MolWt *mean* response from the flattened array while quoting
``delta_mae`` from the hierarchical one.  This stage adds a single table in
which every column states its weighting, and it keeps the original flat
numbers untouched (nothing here overwrites ``partB_*_metrics.json``).

Contents
--------
* ``analysis/weighted_response.csv`` — per task x split x family (+ one POOLED
  row per task/split), molecule-weighted and pair/view-weighted descriptions
  side by side, the arithmetic checks, and the worst single case.
* ``analysis/weighted_response.json`` — the same rows plus metadata: weighting
  definition, target std, source file hashes, the frozen bootstrap reuse, and
  the ``MOLWT`` identity check (``Delta_ref == 0`` => the hierarchical
  ``delta_mae`` must equal the molecule-weighted ``mean |Delta_pred|``).

No model is loaded, no prediction is recomputed, no training and no new
bootstrap design: this stage reads the committed pair tables only and reuses
the round's own frozen scaffold bootstrap draws (seed ``20261114``, 5000
replicates) to attach an interval to one derived descriptive quantity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import round_common as RC

WEIGHT_SCHEME = (
    "molecule -> instance(center,target,control) -> control/donor; "
    "per pair the T and K views are averaged first (0.5*(|dpT|+|dpK|)); "
    "then instance mean, then molecule mean, then an unweighted mean over molecules"
)
FLAT_SCHEME = "pair/view weighted: every (pair, view in {T,K0}) is one observation"

COLUMNS = (
    "task", "split", "family", "scope", "n_pairs", "n_molecules", "n_instances",
    "target_std", "units", "weight_scheme",
    "abs_delta_pred_mean_mol", "abs_delta_pred_median_mol", "abs_delta_pred_p95_mol",
    "abs_delta_pred_max_mol", "abs_delta_pred_mean_mol_over_target_std",
    "abs_delta_pred_mean_mol_ci95_low", "abs_delta_pred_mean_mol_ci95_high",
    "abs_delta_pred_mean_mol_ci95_source",
    "baseline_mean_mol", "delta_mae_mol", "delta_mae_mol_ci95_low", "delta_mae_mol_ci95_high",
    "mean_abs_dpred_minus_dref_mol",
    "abs_delta_pred_mean_pairview", "abs_delta_pred_median_pairview",
    "abs_delta_pred_p95_pairview", "abs_delta_pred_max_pairview",
    "abs_delta_pred_mean_T_pairview", "abs_delta_pred_mean_K_pairview",
    "baseline_mean_pairview",
    "check_delta_mae_vs_committed", "check_molwt_identity",
    "worst_view", "worst_pair_id", "worst_row_id", "worst_abs_dpred",
    "notes",
)


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        s = str(x).strip()
        return float(s) if s not in ("", "nan", "None") else d
    except Exception:
        return d


def _sha(path: Path) -> str:
    return RC.sha256_file(path)


def _scaffold() -> dict[int, str]:
    raw_rows = RC.SC.load_raw_rows()["rows"]["Lipophilicity"]
    return {rid: RC.G.bemis_murcko_key(str(raw_rows[rid]["smiles"])) for rid in range(RC.SC.N_MOLECULES)}


def _multiplicity(split: str, scaffold: Mapping[int, str]) -> dict[str, Any]:
    """Exactly the frozen draws used by ``evaluate.py`` (same seed, same rows)."""
    rng = np.random.default_rng(RC.BOOTSTRAP_SEED)
    rows = [int(x) for x in RC.SC.load_source("none")["rows"][split]]
    return RC.AN.cluster_multiplicity(rows, scaffold, rng=rng, replicates=RC.BOOTSTRAP_REPLICATES)


def _quantiles(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return (float("nan"),) * 3
    return (float(np.median(values)), float(np.percentile(values, 95)), float(values.max()))


def _block(
    task: str,
    split: str,
    family: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Mapping[str, Any],
    target_std: float,
    units: str,
    mult: Mapping[str, Any],
    family_key: str | None,
) -> dict[str, Any]:
    for r in rows:  # per-pair mean of the two views (same shape as abs_err_pair)
        r["abs_dpred_pair"] = 0.5 * (abs(_f(r["abs_dpred_T"])) + abs(_f(r["abs_dpred_K"])))
    mm = RC.AN.molecule_means(rows, ("abs_dpred_pair", "abs_err_pair", "baseline_pair"))
    mol_abs = np.asarray([mm[k]["abs_dpred_pair"] for k in mm], dtype=np.float64)
    med, p95, mx = _quantiles(mol_abs)
    mean_mol = float(mol_abs.mean()) if mol_abs.size else float("nan")
    delta_mae = RC.AN.point_mean({k: mm[k]["abs_err_pair"] for k in mm})
    baseline_mol = RC.AN.point_mean({k: mm[k]["baseline_pair"] for k in mm})

    reps_abs = RC.AN.bootstrap_mean(mult, {k: mm[k]["abs_dpred_pair"] for k in mm})
    reps_err = RC.AN.bootstrap_mean(mult, {k: mm[k]["abs_err_pair"] for k in mm})
    ci_abs = RC.AN.pct_ci(reps_abs)
    ci_err = RC.AN.pct_ci(reps_err)

    # committed (already reported) values, read back for the cross-check
    committed_delta_mae = float("nan")
    committed_ci = [float("nan"), float("nan")]
    committed_flat: dict[str, float] = {}
    if family_key is not None:
        blk = metrics["family_stats"].get(f"{task}/{split}/{family_key}")
        if blk and "delta_mae" in blk:
            committed_delta_mae = float(blk["delta_mae"])
            committed_ci = [float(x) for x in blk["delta_mae_ci95"]]
            committed_flat = {k: float(v) for k, v in blk["abs_delta_pred"].items()}

    # flat (pair/view) description, recomputed from the same rows
    flat = np.asarray(
        [abs(_f(r["abs_dpred_T"])) for r in rows] + [abs(_f(r["abs_dpred_K"])) for r in rows],
        dtype=np.float64,
    )
    flat_base = np.asarray(
        [abs(_f(r["baseline_T"])) for r in rows] + [abs(_f(r["baseline_K"])) for r in rows],
        dtype=np.float64,
    )
    flat_T = np.asarray([abs(_f(r["abs_dpred_T"])) for r in rows], dtype=np.float64)
    flat_K = np.asarray([abs(_f(r["abs_dpred_K"])) for r in rows], dtype=np.float64)
    fmed, fp95, fmx = _quantiles(flat)

    # worst single view in this cell (kept, never trimmed from the statistics)
    worst = (-1.0, "", "", "")
    for r in rows:
        for v in ("T", "K"):
            a = abs(_f(r[f"abs_dpred_{v}"]))
            if a > worst[0]:
                worst = (a, str(r["pair_id"]), str(r["row_id"]), v)

    n_inst = len({(int(r["center"]), int(r["target"]), int(r["control"])) for r in rows})

    check_delta_mae = (
        abs(delta_mae - committed_delta_mae) <= 1e-9 if np.isfinite(committed_delta_mae) else None
    )
    # MOLWT: Delta_ref == 0 for every pair => delta_mae and mean|Delta_pred| are the
    # same arithmetic quantity.  Reported for both tasks (WC_LOGP: must be False).
    molwt_identity = abs(mean_mol - delta_mae) <= 1e-12

    notes = []
    if family_key is None:
        notes.append(
            "POOLED: all five families of this split, main CF only; molecules shared across "
            "families are counted once (their instances pooled), so this is NOT the mean of family means"
        )
    else:
        notes.append("main CF only (O, T and K0 all absent from the Train original set)")
    notes.append(f"n_pairs={len(rows)} unique pair_id; n_molecules={len(mm)} unique row_id; n_instances={n_inst}")
    notes.append("no dedup of views for the statistics; unique views are deduplicated only for computation")
    if task == "molwt":
        notes.append("Delta_ref == 0 by construction; the zero-change baseline is exactly 0, so gain = -delta_mae is not a quality signal")
    if task == "wc_logp":
        notes.append("units = Wildman-Crippen logP; a computational reference, not experimental logD")
    if family_key == "ester":
        notes.append("ester: 8 molecules / 46 Test pairs; unresolved (interval covers 0)")
    if family_key == "sulfonate":
        notes.append("sulfonate: no Valid/Test rewire and no C; untested, never 'zero'")

    return {
        "task": task,
        "split": split,
        "family": family,
        "scope": "FULL main_cf",
        "n_pairs": len(rows),
        "n_molecules": len(mm),
        "n_instances": n_inst,
        "target_std": target_std,
        "units": units,
        "weight_scheme": WEIGHT_SCHEME,
        "abs_delta_pred_mean_mol": mean_mol,
        "abs_delta_pred_median_mol": med,
        "abs_delta_pred_p95_mol": p95,
        "abs_delta_pred_max_mol": mx,
        "abs_delta_pred_mean_mol_over_target_std": mean_mol / target_std,
        "abs_delta_pred_mean_mol_ci95_low": ci_abs[0],
        "abs_delta_pred_mean_mol_ci95_high": ci_abs[1],
        "abs_delta_pred_mean_mol_ci95_source": (
            "frozen scaffold bootstrap (seed %d, %d replicates, same draws as evaluate.py); "
            "identical to delta_mae_ci95 when Delta_ref == 0" % (RC.BOOTSTRAP_SEED, RC.BOOTSTRAP_REPLICATES)
        ),
        "baseline_mean_mol": baseline_mol,
        "delta_mae_mol": delta_mae,
        "delta_mae_mol_ci95_low": ci_err[0],
        "delta_mae_mol_ci95_high": ci_err[1],
        "mean_abs_dpred_minus_dref_mol": delta_mae,
        "abs_delta_pred_mean_pairview": float(flat.mean()) if flat.size else float("nan"),
        "abs_delta_pred_median_pairview": fmed,
        "abs_delta_pred_p95_pairview": fp95,
        "abs_delta_pred_max_pairview": fmx,
        "abs_delta_pred_mean_T_pairview": float(flat_T.mean()) if flat_T.size else float("nan"),
        "abs_delta_pred_mean_K_pairview": float(flat_K.mean()) if flat_K.size else float("nan"),
        "baseline_mean_pairview": float(flat_base.mean()) if flat_base.size else float("nan"),
        "check_delta_mae_vs_committed": check_delta_mae,
        "check_molwt_identity": bool(molwt_identity),
        "worst_view": worst[3],
        "worst_pair_id": worst[1],
        "worst_row_id": worst[2],
        "worst_abs_dpred": worst[0],
        "notes": " | ".join(notes),
        "_committed_delta_mae": committed_delta_mae,
        "_committed_delta_mae_ci95": committed_ci,
        "_committed_flat": committed_flat,
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="valid,test")
    args = ap.parse_args(argv)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]

    scaffold = _scaffold()
    mult = {s: _multiplicity(s, scaffold) for s in splits}

    rows_out: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "round_id": RC.ROUND_ID,
        "generated_utc": RC.utcnow(),
        "git_commit": RC.git_commit(),
        "git_dirty": RC.git_dirty(),
        "weight_scheme_molecule": WEIGHT_SCHEME,
        "weight_scheme_pairview": FLAT_SCHEME,
        "why": (
            "partB_*_metrics.json stores delta_mae (molecule->instance->control->donor) and "
            "abs_delta_pred.* (flattened T/K views) side by side; the two are not the same "
            "weighting.  This table states both explicitly and keeps the original numbers "
            "untouched."
        ),
        "new_training": 0,
        "new_model_inference": 0,
        "new_interventions": 0,
        "gpu_hours": 0.0,
        "inputs": {},
        "tasks": {},
        "checks": {},
    }

    for task in RC.TASKS:
        metrics = RC.read_json(RC.NEW_ANALYSIS / f"partB_{task}_metrics.json")
        pairs_path = RC.NEW_ANALYSIS / f"partB_{task}_pairs.csv"
        pairs = RC.read_csv(pairs_path)
        target_std = float(metrics["target_norm"]["scale"])
        units = str(metrics["units"])
        meta["inputs"][f"partB_{task}_pairs.csv"] = _sha(pairs_path)
        meta["inputs"][f"partB_{task}_metrics.json"] = _sha(RC.NEW_ANALYSIS / f"partB_{task}_metrics.json")
        meta["tasks"][task] = {
            "units": units,
            "target_norm": metrics["target_norm"],
            "original_graph_test_rmse": metrics["original_graph_metrics"]["test"]["rmse"],
            "original_graph_test_r2": metrics["original_graph_metrics"]["test"]["r2"],
        }

        task_rows: dict[str, Any] = {}
        for split in splits:
            sel = [dict(r) for r in pairs if r["split"] == split and str(r["main_cf"]) == "1"]
            if not sel:
                continue
            fams = [f for f in RC.SC.FAMILIES if any(r["family"] == f for r in sel)]
            for fam in fams + ["POOLED"]:
                sub = sel if fam == "POOLED" else [r for r in sel if r["family"] == fam]
                blk = _block(
                    task, split, fam, sub,
                    metrics=metrics, target_std=target_std, units=units,
                    mult=mult[split], family_key=None if fam == "POOLED" else fam,
                )
                rows_out.append(blk)
                task_rows[f"{split}/{fam}"] = blk

        meta["tasks"][task]["cells"] = len(task_rows)

    # ---- checks -----------------------------------------------------------
    ident = [r for r in rows_out if r["task"] == "molwt"]
    wc = [r for r in rows_out if r["task"] == "wc_logp"]
    meta["checks"]["molwt_identity_all_cells"] = bool(all(r["check_molwt_identity"] for r in ident))
    meta["checks"]["molwt_identity_max_abs_gap"] = float(
        max(abs(r["abs_delta_pred_mean_mol"] - r["delta_mae_mol"]) for r in ident)
    )
    meta["checks"]["wc_logp_identity_holds"] = bool(any(r["check_molwt_identity"] for r in wc))
    meta["checks"]["delta_mae_matches_committed_all_cells"] = bool(
        all(r["check_delta_mae_vs_committed"] in (True, None) for r in rows_out)
    )
    meta["checks"]["delta_mae_max_abs_gap_vs_committed"] = float(
        max(abs(r["delta_mae_mol"] - r["_committed_delta_mae"]) for r in rows_out)
    )
    meta["checks"]["delta_mae_ci95_reproduced_max_abs_gap"] = float(
        max(
            max(abs(r["delta_mae_mol_ci95_low"] - r["_committed_delta_mae_ci95"][0]),
                abs(r["delta_mae_mol_ci95_high"] - r["_committed_delta_mae_ci95"][1]))
            for r in rows_out
        )
    )
    meta["checks"]["flat_block_reproduced_max_abs_gap"] = float(
        max(
            (
                max(abs(r["abs_delta_pred_mean_pairview"] - r["_committed_flat"]["mean"]),
                    abs(r["abs_delta_pred_median_pairview"] - r["_committed_flat"]["median"]),
                    abs(r["abs_delta_pred_p95_pairview"] - r["_committed_flat"]["p95"]),
                    abs(r["abs_delta_pred_max_pairview"] - r["_committed_flat"]["max"]))
                if r["_committed_flat"] else 0.0
            )
            for r in rows_out
        )
    )
    # the previously quoted pooled MolWt/WC_LOGP numbers are T-view only
    meta["checks"]["legacy_pooled_numbers_are_T_view_only"] = {
        "note": (
            "REPORT 3.3/3.4 quoted pooled Test mean |dpred| 0.246 / mean |dref| 0.241 / "
            "mean |dpred-dref| 0.072 and MolWt 1.81 g/mol; these are the T-view, pair-weighted "
            "means, not the T+K average and not molecule weighted"
        ),
        "wc_logp_test_POOLED_T_only_mean_abs_dpred": next(
            r["abs_delta_pred_mean_T_pairview"] for r in wc if r["split"] == "test" and r["family"] == "POOLED"
        ),
        "wc_logp_test_POOLED_mean_abs_dpred_TK_pairview": next(
            r["abs_delta_pred_mean_pairview"] for r in wc if r["split"] == "test" and r["family"] == "POOLED"
        ),
        "wc_logp_test_POOLED_mean_abs_dpred_molecule": next(
            r["abs_delta_pred_mean_mol"] for r in wc if r["split"] == "test" and r["family"] == "POOLED"
        ),
        "molwt_test_POOLED_T_only_mean_abs_dpred": next(
            r["abs_delta_pred_mean_T_pairview"] for r in ident if r["split"] == "test" and r["family"] == "POOLED"
        ),
        "molwt_test_POOLED_mean_abs_dpred_TK_pairview": next(
            r["abs_delta_pred_mean_pairview"] for r in ident if r["split"] == "test" and r["family"] == "POOLED"
        ),
        "molwt_test_POOLED_mean_abs_dpred_molecule": next(
            r["abs_delta_pred_mean_mol"] for r in ident if r["split"] == "test" and r["family"] == "POOLED"
        ),
    }

    for r in rows_out:
        r.pop("_committed_delta_mae", None)
        r.pop("_committed_delta_mae_ci95", None)
        r.pop("_committed_flat", None)

    RC.write_csv(RC.NEW_ANALYSIS / "weighted_response.csv", rows_out, COLUMNS)
    meta["files"] = {"weighted_response.csv": _sha(RC.NEW_ANALYSIS / "weighted_response.csv")}
    RC.write_json(RC.NEW_ANALYSIS / "weighted_response.json", meta)
    RC.log(
        "weighted_response: %d rows; molwt identity=%s; delta_mae gap vs committed=%.2e"
        % (len(rows_out), meta["checks"]["molwt_identity_all_cells"],
           meta["checks"]["delta_mae_max_abs_gap_vs_committed"])
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
