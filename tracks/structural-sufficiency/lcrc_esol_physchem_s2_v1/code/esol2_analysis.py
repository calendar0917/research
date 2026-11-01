"""Validation-only comparison and uncertainty analysis for the four arms.

Nothing here touches Test: the previous ESOL round already read and analysed the
old Test split, so it is not a blind set for this round.  Everything is computed
from the frozen BASE runs (read-only) and the new Train/Validation runs.

Outputs
-------
``analysis/valid_comparison.csv``   all six runs + the three reused BASE runs
``analysis/paired_valid.csv``       same-seed paired differences
``analysis/paired_summary.json``    mean/sd, win directions, group bootstrap,
                                    scaffold-group contributions
``analysis/physchem_usage.json``    descriptor-usage diagnostic (no training)
"""

from __future__ import annotations

import csv

from typing import Any, Mapping, Sequence

import numpy as np

import esol2_common as C
import esol2_data as D
import esol2_models as M
import esol2_train as T

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261103


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def _record_from_train_json(payload: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    train_rmse = float(payload["metrics"]["train"]["rmse"])
    valid_rmse = float(payload["metrics"]["valid"]["rmse"])
    return {
        "model": str(payload["family"]),
        "model_display": str(payload["model"]),
        "seed": int(payload["seed"]),
        "dropout": float(payload["model_audit"]["dropout"]),
        "params_nominal": int(payload["model_audit"]["parameters"]),
        "params_effective": int(payload["model_audit"].get("parameters_effective", payload["model_audit"]["parameters"])),
        "best_epoch": int(payload["checkpoint"]["best_epoch"]),
        "epochs_run": int(payload["checkpoint"]["epochs_run"]),
        "train_rmse": train_rmse,
        "train_mae": float(payload["metrics"]["train"]["mae"]),
        "valid_rmse": valid_rmse,
        "valid_mae": float(payload["metrics"]["valid"]["mae"]),
        "valid_r2": float(payload["metrics"]["valid"]["r2"]),
        "train_valid_gap": float(valid_rmse - train_rmse),
        "checkpoint_sha256": str(payload["checkpoint"]["sha256"]),
        "source": source,
        "run_dir": C.track_rel(C.v1_run_dir(payload["arm"]) if source == "v1_readonly" else C.run_dir(payload["arm"])),
    }


def base_records() -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for seed in C.SEEDS:
        arm = f"base_s{seed}"
        payload = C.read_json(C.v1_run_dir(arm) / "train.json")
        if payload.get("test_labels_read") is not False:
            raise PermissionError(f"{arm}: v1 record claims Test-label access")
        out[seed] = _record_from_train_json(payload, source="v1_readonly")
    return out


def new_records() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm in C.arms_with_runs():
        payload = C.read_json(C.run_dir(arm) / "train.json")
        if payload.get("smoke") is not False:
            raise PermissionError(f"{arm}: refusing smoke run in the comparison")
        out[arm] = _record_from_train_json(payload, source="new_run")
    return out


def _valid_predictions(arm: str, source: str, views: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-derive Validation predictions from the **committed checkpoint**.

    The ``*.npz`` prediction files are git-ignored, so nothing here depends on
    them; when a local copy exists it is used only as a cross-check.
    """
    run_dir = C.v1_run_dir(arm) if source == "v1_readonly" else C.run_dir(arm)
    train_json = C.read_json(run_dir / "train.json")
    model_kind, seed = C.arm_parts(arm)
    model, _meta = M.build(
        model_kind,
        views["spec"],
        views["train_norm"],
        seed,
        "cpu",
        dropout=float(C.MODEL_DROPOUT[model_kind]),
        physchem_norm=views.get("physchem_norm"),
    )
    M.load_checkpoint(
        model,
        run_dir / "checkpoint.pt",
        expected_sha256=train_json["checkpoint"]["sha256"],
        expected_family=model_kind,
        expected_seed=seed,
    )
    _train, physchem_valid = D.physchem_for(model_kind, views)
    pred = T.denormalise(
        M.predict(model, views["valid_cache"], "cpu", physchem=physchem_valid), views["target_norm"]
    )
    y = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    recorded = float(train_json["metrics"]["valid"]["rmse"])
    recomputed = float(C.all_metrics(pred, y)["rmse"])
    if abs(recomputed - recorded) > 1.0e-6:
        raise RuntimeError(f"{arm}: recomputed Validation RMSE {recomputed} != recorded {recorded}")
    rows = np.asarray(views["split"]["rows"]["valid"], dtype=np.int64)
    npz_path = run_dir / "valid_predictions.npz"
    if npz_path.exists():
        saved = np.load(npz_path)
        if not np.array_equal(np.asarray(saved["rows"], dtype=np.int64), rows):
            raise PermissionError(f"{arm}: saved Validation rows are not the frozen split order")
        # v1 stored its Validation predictions from a CUDA forward; a CPU replay
        # differs only by float32 accumulation order (~1e-6), so this is a loose
        # cross-check, not the primary assertion (that is the RMSE above).
        if float(np.max(np.abs(np.asarray(saved["pred"], dtype=np.float64) - pred))) > 1.0e-4:
            raise RuntimeError(f"{arm}: saved predictions disagree with the checkpoint replay")
    return rows, pred, y


# ---------------------------------------------------------------------------
# scaffold groups (same policy as the frozen splitter)
# ---------------------------------------------------------------------------


def valid_group_keys(views: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    scaffolds = C.V1C.read_json(C.SCAFFOLD_FILE)["scaffold_by_row"]
    smiles = views["split"]["smiles"]
    rows = np.asarray(views["split"]["rows"]["valid"], dtype=np.int64)
    keys = [
        str(scaffolds[int(row)]) if str(scaffolds[int(row)]) != "" else f"__ACYCLIC__{smiles[int(row)]}"
        for row in rows
    ]
    return keys, rows


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err)))) if err.size else float("nan")


def _paired_group_bootstrap(
    err_base: np.ndarray, err_cand: np.ndarray, group_index: Sequence[Sequence[int]], *, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_groups = len(group_index)
    deltas = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for i in range(BOOTSTRAP_RESAMPLES):
        pick = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([np.asarray(group_index[g], dtype=np.int64) for g in pick])
        deltas[i] = _rmse(err_cand[idx]) - _rmse(err_base[idx])
    return {
        "resamples": int(BOOTSTRAP_RESAMPLES),
        "seed": int(seed),
        "observed_delta": float(_rmse(err_cand) - _rmse(err_base)),
        "mean_delta": float(deltas.mean()),
        "sd_delta": float(deltas.std(ddof=1)),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "fraction_resamples_candidate_better": float((deltas < 0).mean()),
        "unit": "frozen Bemis-Murcko scaffold group (acyclic = SMILES singleton), paired resampling",
        "note": (
            "descriptive only: it propagates within-this-partition group correlation, "
            "not uncertainty across different scaffold partitions"
        ),
    }


def _group_contributions(
    err_base: np.ndarray, err_cand: np.ndarray, keys: Sequence[str], rows: np.ndarray
) -> list[dict[str, Any]]:
    order = sorted(set(keys))
    sse_base = np.square(err_base)
    sse_cand = np.square(err_cand)
    total_delta = float(sse_cand.sum() - sse_base.sum())
    recs = []
    for key in order:
        mask = np.asarray([k == key for k in keys])
        if not mask.any():
            continue
        recs.append(
            {
                "group": key,
                "n": int(mask.sum()),
                "csv_rows": [int(x) for x in rows[mask]],
                "rmse_base": _rmse(err_base[mask]),
                "rmse_candidate": _rmse(err_cand[mask]),
                "delta_rmse": float(_rmse(err_cand[mask]) - _rmse(err_base[mask])),
                "delta_sse": float(sse_cand[mask].sum() - sse_base[mask].sum()),
                "share_of_total_delta_sse": float(
                    (sse_cand[mask].sum() - sse_base[mask].sum()) / total_delta if total_delta != 0.0 else 0.0
                ),
                "share_of_base_sse": float(sse_base[mask].sum() / sse_base.sum()),
            }
        )
    recs.sort(key=lambda rec: abs(rec["delta_sse"]), reverse=True)
    return recs


def _concentration(err: np.ndarray) -> dict[str, float]:
    sse = np.sort(np.square(err))[::-1]
    total = float(sse.sum())
    return {
        "top1_molecule_share": float(sse[0] / total) if total else 0.0,
        "top1pct_share": float(sse[: max(1, int(round(0.01 * sse.size)))].sum() / total) if total else 0.0,
        "top5pct_share": float(sse[: max(1, int(round(0.05 * sse.size)))].sum() / total) if total else 0.0,
        "top10pct_share": float(sse[: max(1, int(round(0.10 * sse.size)))].sum() / total) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# main comparison
# ---------------------------------------------------------------------------


def analyse() -> dict[str, Any]:
    views = D.build_views()
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    keys, rows = valid_group_keys(views)
    group_index = [np.flatnonzero(np.asarray(keys) == key) for key in sorted(set(keys))]

    base = base_records()
    new = new_records()
    missing = [C.arm_key(model, seed) for model in C.CANDIDATES for seed in C.SEEDS if C.arm_key(model, seed) not in new]
    if missing:
        raise PermissionError(f"missing new runs: {missing}")

    all_records: list[dict[str, Any]] = []
    for seed in C.SEEDS:
        all_records.append(base[seed])
    for arm in C.NEW_ARMS:
        if arm in new:
            all_records.append(new[arm])

    with open(C.VALID_COMPARISON, "w", newline="", encoding="utf-8") as handle:
        fields = [
            "model",
            "model_display",
            "seed",
            "dropout",
            "params_nominal",
            "params_effective",
            "best_epoch",
            "epochs_run",
            "train_rmse",
            "train_mae",
            "valid_rmse",
            "valid_mae",
            "valid_r2",
            "train_valid_gap",
            "source",
            "checkpoint_sha256",
            "run_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in all_records:
            writer.writerow({key: rec[key] for key in fields})

    summary: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "n_valid": int(y_valid.size),
        "n_valid_groups": int(len(group_index)),
        "metric_unit": "raw log(mol/L)",
        "test_evaluated": False,
        "validation_caveat": (
            "the 113-molecule Validation split has been used by several rounds; it is not a "
            "fresh confirmation set, and molecule errors are correlated inside scaffold groups"
        ),
        "base": {
            "valid_rmse_mean": float(np.mean([base[s]["valid_rmse"] for s in C.SEEDS])),
            "valid_rmse_sd": float(np.std([base[s]["valid_rmse"] for s in C.SEEDS], ddof=1)),
            "valid_mae_mean": float(np.mean([base[s]["valid_mae"] for s in C.SEEDS])),
            "training_rmse_mean": float(np.mean([base[s]["train_rmse"] for s in C.SEEDS])),
        },
        "candidates": {},
    }

    paired_rows: list[dict[str, Any]] = []
    for model in C.CANDIDATES:
        cand_records = {seed: new[C.arm_key(model, seed)] for seed in C.SEEDS}
        per_seed = []
        for seed in C.SEEDS:
            b, c = base[seed], cand_records[seed]
            per_seed.append(
                {
                    "seed": int(seed),
                    "base_valid_rmse": b["valid_rmse"],
                    "candidate_valid_rmse": c["valid_rmse"],
                    "delta_valid_rmse": float(c["valid_rmse"] - b["valid_rmse"]),
                    "base_valid_mae": b["valid_mae"],
                    "candidate_valid_mae": c["valid_mae"],
                    "delta_valid_mae": float(c["valid_mae"] - b["valid_mae"]),
                    "base_train_rmse": b["train_rmse"],
                    "candidate_train_rmse": c["train_rmse"],
                    "delta_train_rmse": float(c["train_rmse"] - b["train_rmse"]),
                    "base_best_epoch": b["best_epoch"],
                    "candidate_best_epoch": c["best_epoch"],
                    "delta_best_epoch": int(c["best_epoch"] - b["best_epoch"]),
                    "delta_train_valid_gap": float(c["train_valid_gap"] - b["train_valid_gap"]),
                }
            )
            paired_rows.append(
                {
                    "candidate": C.MODEL_DISPLAY[model],
                    "family": model,
                    "seed": int(seed),
                    "base_valid_rmse": b["valid_rmse"],
                    "candidate_valid_rmse": c["valid_rmse"],
                    "delta_valid_rmse": float(c["valid_rmse"] - b["valid_rmse"]),
                    "base_valid_mae": b["valid_mae"],
                    "candidate_valid_mae": c["valid_mae"],
                    "delta_valid_mae": float(c["valid_mae"] - b["valid_mae"]),
                    "base_train_rmse": b["train_rmse"],
                    "candidate_train_rmse": c["train_rmse"],
                    "delta_train_rmse": float(c["train_rmse"] - b["train_rmse"]),
                    "base_best_epoch": b["best_epoch"],
                    "candidate_best_epoch": c["best_epoch"],
                    "delta_best_epoch": int(c["best_epoch"] - b["best_epoch"]),
                    "candidate_better": bool(c["valid_rmse"] < b["valid_rmse"]),
                }
            )

        deltas = np.asarray([rec["delta_valid_rmse"] for rec in per_seed], dtype=np.float64)
        delta_mae = np.asarray([rec["delta_valid_mae"] for rec in per_seed], dtype=np.float64)
        cand_rmse = np.asarray([rec["candidate_valid_rmse"] for rec in per_seed], dtype=np.float64)
        base_rmse = np.asarray([rec["base_valid_rmse"] for rec in per_seed], dtype=np.float64)

        # bootstrap + contributions, per seed and pooled over the three seeds
        bootstrap: dict[str, Any] = {}
        group_contrib: dict[str, Any] = {}
        molecule_direction: dict[str, Any] = {}
        for seed in C.SEEDS:
            _rows, pred_base, y = _valid_predictions(f"base_s{seed}", "v1_readonly", views)
            _rows, pred_cand, y_cand = _valid_predictions(C.arm_key(model, seed), "new_run", views)
            if not np.array_equal(y, y_cand):
                raise PermissionError("Validation labels differ between runs")
            err_base = pred_base - y
            err_cand = pred_cand - y
            bootstrap[f"s{seed}"] = _paired_group_bootstrap(err_base, err_cand, group_index, seed=BOOTSTRAP_SEED + seed)
            contrib = _group_contributions(err_base, err_cand, keys, rows)
            group_contrib[f"s{seed}"] = {
                "top_groups_by_abs_delta_sse": contrib[:5],
                "n_groups": len(contrib),
                "top1_group_abs_share_of_delta_sse": abs(contrib[0]["share_of_total_delta_sse"]) if contrib else 0.0,
                "top3_groups_abs_share_of_delta_sse": float(
                    sum(abs(rec["share_of_total_delta_sse"]) for rec in contrib[:3])
                ),
                "n_groups_for_50pct_of_abs_delta_sse": _groups_for_half(contrib),
                "base_error_concentration": _concentration(err_base),
                "candidate_error_concentration": _concentration(err_cand),
            }
            molecule_direction[f"s{seed}"] = {
                "n_improved": int((np.abs(err_cand) < np.abs(err_base)).sum()),
                "n_worse": int((np.abs(err_cand) > np.abs(err_base)).sum()),
                "n_unchanged": int((np.abs(err_cand) == np.abs(err_base)).sum()),
                "delta_rmse": float(_rmse(err_cand) - _rmse(err_base)),
            }
        # pooled over seeds: average of the three per-seed bootstrap means
        pooled_ci = {
            "mean_delta": float(np.mean([bootstrap[f"s{s}"]["mean_delta"] for s in C.SEEDS])),
            "ci95_low": float(np.mean([bootstrap[f"s{s}"]["ci95_low"] for s in C.SEEDS])),
            "ci95_high": float(np.mean([bootstrap[f"s{s}"]["ci95_high"] for s in C.SEEDS])),
            "fraction_resamples_candidate_better": float(
                np.mean([bootstrap[f"s{s}"]["fraction_resamples_candidate_better"] for s in C.SEEDS])
            ),
            "note": "mean of the three per-seed group-bootstrap intervals (descriptive)",
        }

        summary["candidates"][model] = {
            "model_display": C.MODEL_DISPLAY[model],
            "params_nominal": cand_records[C.SEEDS[0]]["params_nominal"],
            "params_effective": cand_records[C.SEEDS[0]]["params_effective"],
            "dropout": cand_records[C.SEEDS[0]]["dropout"],
            "per_seed": per_seed,
            "delta_valid_rmse_mean": float(deltas.mean()),
            "delta_valid_rmse_sd": float(deltas.std(ddof=1)),
            "delta_valid_mae_mean": float(delta_mae.mean()),
            "delta_valid_mae_sd": float(delta_mae.std(ddof=1)),
            "delta_train_rmse_mean": float(np.mean([rec["delta_train_rmse"] for rec in per_seed])),
            "delta_best_epoch_mean": float(np.mean([rec["delta_best_epoch"] for rec in per_seed])),
            "n_seeds_candidate_better": int((deltas < 0).sum()),
            "n_seeds_candidate_worse": int((deltas > 0).sum()),
            "n_seeds_tied": int((np.abs(deltas) <= 1e-9).sum()),
            "base_valid_rmse_mean": float(base_rmse.mean()),
            "base_valid_rmse_sd": float(base_rmse.std(ddof=1)),
            "candidate_valid_rmse_mean": float(cand_rmse.mean()),
            "candidate_valid_rmse_sd": float(cand_rmse.std(ddof=1)),
            "candidate_valid_mae_mean": float(np.mean([rec["candidate_valid_mae"] for rec in per_seed])),
            "candidate_train_rmse_mean": float(np.mean([rec["candidate_train_rmse"] for rec in per_seed])),
            "candidate_best_epoch_mean": float(np.mean([rec["candidate_best_epoch"] for rec in per_seed])),
            "group_bootstrap": bootstrap,
            "group_bootstrap_pooled": pooled_ci,
            "group_contributions": group_contrib,
            "molecule_direction": molecule_direction,
            "sign_convention": "delta = candidate - base; negative = candidate better",
        }

    with open(C.PAIRED_VALID, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    # per-molecule Validation table for all nine runs (committed CSV artifact)
    columns: dict[str, np.ndarray] = {"csv_row": rows, "y_true": y_valid, "group": np.asarray(keys)}
    for seed in C.SEEDS:
        columns[f"base_s{seed}"] = _valid_predictions(f"base_s{seed}", "v1_readonly", views)[1]
    for arm in C.NEW_ARMS:
        columns[arm] = _valid_predictions(arm, "new_run", views)[1]
    with open(C.ANALYSIS / "per_molecule_valid.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(columns))
        for i in range(int(rows.size)):
            writer.writerow([columns[name][i] for name in columns])

    summary["physchem_usage"] = descriptor_usage(views)
    C.write_json(C.PAIRED_SUMMARY, summary)
    C.write_json(C.PHYSCHEM_USAGE, summary["physchem_usage"])
    return summary


def _groups_for_half(contrib: Sequence[Mapping[str, Any]]) -> int:
    total = sum(abs(rec["delta_sse"]) for rec in contrib)
    if total == 0.0:
        return 0
    cumulative = 0.0
    for i, rec in enumerate(contrib, start=1):
        cumulative += abs(rec["delta_sse"])
        if cumulative >= 0.5 * total:
            return i
    return len(contrib)


# ---------------------------------------------------------------------------
# descriptor-usage diagnostic (no training, no selection)
# ---------------------------------------------------------------------------


def descriptor_usage(views: Mapping[str, Any]) -> dict[str, Any]:
    """Replace Validation descriptors by the Train mean and compare predictions."""
    spec = views["spec"]
    physchem_valid = np.asarray(views["physchem_valid"], dtype=np.float64)
    mean = np.asarray(views["physchem_norm"]["mean"], dtype=np.float64)
    neutral = np.repeat(mean.reshape(1, -1), physchem_valid.shape[0], axis=0)
    y = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    norm = views["target_norm"]
    out: dict[str, Any] = {
        "question": "does the trained PHYSCHEM6 use the added descriptors?",
        "intervention": "Validation descriptors set to the Train mean (standardised value 0); no retraining",
        "caveat": "a descriptive usage check, not a causal claim and not a configuration signal",
        "per_seed": {},
    }
    for seed in C.SEEDS:
        arm = C.arm_key("physchem6", seed)
        run_dir = C.run_dir(arm)
        if not (run_dir / "train.json").exists():
            continue
        train_json = C.read_json(run_dir / "train.json")
        model, _meta = M.build_physchem6(
            spec,
            views["train_norm"],
            seed,
            "cpu",
            0.10,
            physchem_norm=views["physchem_norm"],
        )
        M.load_checkpoint(
            model,
            run_dir / "checkpoint.pt",
            expected_sha256=train_json["checkpoint"]["sha256"],
            expected_family="physchem6",
            expected_seed=seed,
        )
        pred_true = T.denormalise(M.predict(model, views["valid_cache"], "cpu", physchem=physchem_valid), norm)
        pred_neutral = T.denormalise(M.predict(model, views["valid_cache"], "cpu", physchem=neutral), norm)
        delta = pred_neutral - pred_true
        out["per_seed"][f"s{seed}"] = {
            "valid_rmse_with_true_descriptors": float(C.all_metrics(pred_true, y)["rmse"]),
            "valid_rmse_with_train_mean_descriptors": float(C.all_metrics(pred_neutral, y)["rmse"]),
            "mean_abs_prediction_change": float(np.mean(np.abs(delta))),
            "max_abs_prediction_change": float(np.max(np.abs(delta))),
            "median_abs_prediction_change": float(np.median(np.abs(delta))),
            "n_molecules_changed_gt_0.01": int((np.abs(delta) > 0.01).sum()),
            "corr_prediction_change_with_label": float(np.corrcoef(delta, y)[0, 1]),
            "descriptor_role": "rho only",
        }
    values = [rec["mean_abs_prediction_change"] for rec in out["per_seed"].values()]
    out["summary"] = {
        "mean_abs_prediction_change_across_seeds": float(np.mean(values)) if values else None,
        "descriptors_are_used": bool(values and min(values) > 1.0e-4),
    }
    return out


__all__ = ["analyse", "descriptor_usage", "base_records", "new_records", "valid_group_keys"]
