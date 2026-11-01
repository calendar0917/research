"""Checkpoint verification, one-shot Test evaluation and tables for ESOL.

Order is enforced by code, not by convention:

1. ``verify_arm`` — reload each checkpoint, recompute the inner-Validation RMSE
   from scratch, check the curve argmin, the state hash and the *absence* of any
   Test access record;
2. ``freeze_manifest`` — write the checkpoint manifest and its marker;
3. ``evaluate_test`` — refuse unless the marker exists; build the Test cache and
   Test label slice only here, predict once, and write every table.

Nothing in this module can change a checkpoint, an epoch or a seed after the
Test metrics are known: the manifest is re-read and re-hashed, and the Test
predictions are only ever produced from files listed in that frozen manifest.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import esol_common as C
import esol_data as D
import esol_models as M
import esol_split as S
import esol_train as T

TOL = 1.0e-5
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261103


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def _physchem(views: Mapping[str, Any], model_kind: str) -> np.ndarray | None:
    return T._physchem_matrix(views, model_kind)


def verify_arm(arm: str, views: Mapping[str, Any]) -> dict[str, Any]:
    model_kind, seed = C.arm_parts(arm)
    run_path = C.run_dir(arm)
    train_json = C.read_json(run_path / "train.json")
    if train_json.get("smoke") is not False:
        raise PermissionError(f"{arm}: formal verification refuses smoke runs")
    ckpt_path = run_path / "checkpoint.pt"
    ckpt_sha = C.sha256_file(ckpt_path)
    if ckpt_sha != train_json["checkpoint"]["sha256"]:
        raise PermissionError(f"{arm}: checkpoint hash differs from train.json")
    model, _meta = M.build(
        model_kind,
        views["spec"],
        views["train_norm"],
        seed,
        "cpu",
        dropout=float(C.MODEL_DROPOUT[model_kind]),
        physchem_norm=views["physchem_norm"],
    )
    checkpoint = M.load_checkpoint(model, ckpt_path, expected_sha256=ckpt_sha, expected_family=model_kind, expected_seed=seed)
    curve = list(csv.DictReader(open(run_path / "curve.csv", newline="", encoding="utf-8")))
    curve_valid = np.asarray([float(row["valid_rmse_raw"]) for row in curve], dtype=np.float64)
    argmin_epoch = int(np.argmin(curve_valid)) + 1
    if argmin_epoch != int(checkpoint["best_epoch"]):
        raise RuntimeError(f"{arm}: curve argmin {argmin_epoch} != checkpoint best {checkpoint['best_epoch']}")
    physchem = _physchem(views, model_kind)
    pred_valid = T.denormalise(M.predict(model, views["valid_cache"], "cpu", physchem=physchem), views["target_norm"])
    metrics = C.all_metrics(pred_valid, np.asarray(views["labels"]["y_valid"], dtype=np.float64))
    recheck_diff = abs(float(metrics["rmse"]) - float(checkpoint["best_valid_rmse"]))
    if recheck_diff > TOL:
        raise RuntimeError(f"{arm}: inner Validation RMSE does not reproduce ({recheck_diff})")
    state_hash = C.state_dict_fingerprint(checkpoint["state_dict"])
    if state_hash != checkpoint["best_state_hash"]:
        raise RuntimeError(f"{arm}: checkpoint state hash mismatch")
    if train_json.get("test_labels_read") is not False:
        raise PermissionError(f"{arm}: training record claims Test-label access")
    if train_json["data"].get("test_row_values_used") is not False:
        raise PermissionError(f"{arm}: training data record claims Test-row use")
    return {
        "arm": arm,
        "model": C.MODEL_DISPLAY[model_kind],
        "family": model_kind,
        "seed": seed,
        "dropout": float(C.MODEL_DROPOUT[model_kind]),
        "checkpoint": C.track_rel(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_inner_valid_rmse": float(checkpoint["best_valid_rmse"]),
        "best_inner_valid_mae": float(checkpoint["best_valid_mae"]),
        "train_rmse": float(train_json["metrics"]["train"]["rmse"]),
        "train_mae": float(train_json["metrics"]["train"]["mae"]),
        "curve_sha256": C.sha256_file(run_path / "curve.csv"),
        "curve_argmin_epoch": argmin_epoch,
        "reloaded_inner_valid_rmse": float(metrics["rmse"]),
        "reloaded_inner_valid_mae": float(metrics["mae"]),
        "recheck_abs_diff": float(recheck_diff),
        "state_hash": state_hash,
        "parameters": int(train_json["model_audit"]["parameters"]),
        "smoke": False,
    }


def freeze_manifest(arms: Sequence[str]) -> dict[str, Any]:
    views = D.build_views(include_test=False)
    lanes: dict[str, Any] = {}
    for arm in arms:
        if not (C.run_dir(arm) / "train.json").exists():
            raise PermissionError(f"missing ESOL run {arm}")
        lanes[arm] = verify_arm(arm, views)
    body = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "status": "ESOL_MANIFEST_FROZEN_BEFORE_TEST_READ",
        "scope": "one fixed Bemis-Murcko scaffold partition of ESOL (1,128 molecules)",
        "split_manifest": C.track_rel(C.SPLIT_MANIFEST),
        "split_manifest_sha256": C.sha256_file(C.SPLIT_MANIFEST),
        "split_audit_sha256": C.sha256_file(C.SPLIT_AUDIT),
        "config_sha256": C.config_sha256(),
        "code_hashes": C.code_hashes(),
        "n_arms": len(lanes),
        "arms": lanes,
        "test_labels_read_at_freeze_time": False,
        "test_cache_built_at_freeze_time": False,
    }
    body["manifest_sha256"] = C.sha256_json(body)
    C.write_json(C.MANIFEST, body)
    C.write_json(
        C.MANIFEST_FROZEN,
        {
            "manifest_sha256": body["manifest_sha256"],
            "created_utc": body["created_utc"],
            "n_arms": len(lanes),
            "scope": "frozen before the one-shot Test read",
        },
    )
    return body


def _frozen_manifest() -> dict[str, Any]:
    if not C.MANIFEST.exists() or not C.MANIFEST_FROZEN.exists():
        raise PermissionError("ESOL manifest is not frozen; run `manifest` first")
    manifest = C.read_json(C.MANIFEST)
    marker = C.read_json(C.MANIFEST_FROZEN)
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if C.sha256_json(body) != manifest["manifest_sha256"]:
        raise PermissionError("ESOL manifest content changed after the freeze")
    if marker.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise PermissionError("ESOL manifest marker mismatch")
    return manifest


# ---------------------------------------------------------------------------
# one-shot Test evaluation
# ---------------------------------------------------------------------------


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _sse_concentration(err: np.ndarray, frac: float) -> float:
    sq = np.sort(err * err)[::-1]
    k = max(1, int(round(frac * sq.size)))
    total = float(sq.sum())
    return float(sq[:k].sum() / total) if total > 0 else 0.0


def _paired_bootstrap(base_pred: np.ndarray, cand_pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Paired molecule bootstrap for ΔMAE and ΔRMSE (candidate minus BASE)."""
    e_base = base_pred - y
    e_cand = cand_pred - y
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = y.size
    d_mae = np.empty(BOOTSTRAP_RESAMPLES)
    d_rmse = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, size=n)
        d_mae[i] = np.mean(np.abs(e_cand[idx])) - np.mean(np.abs(e_base[idx]))
        d_rmse[i] = np.sqrt(np.mean(e_cand[idx] ** 2)) - np.sqrt(np.mean(e_base[idx] ** 2))
    return {
        "delta_mae_point": float(np.mean(np.abs(e_cand)) - np.mean(np.abs(e_base))),
        "delta_mae_ci95_low": float(np.quantile(d_mae, 0.025)),
        "delta_mae_ci95_high": float(np.quantile(d_mae, 0.975)),
        "delta_mae_ci_crosses_zero": bool(np.quantile(d_mae, 0.025) <= 0.0 <= np.quantile(d_mae, 0.975)),
        "delta_rmse_point": float(np.sqrt(np.mean(e_cand ** 2)) - np.sqrt(np.mean(e_base ** 2))),
        "delta_rmse_ci95_low": float(np.quantile(d_rmse, 0.025)),
        "delta_rmse_ci95_high": float(np.quantile(d_rmse, 0.975)),
        "delta_rmse_ci_crosses_zero": bool(np.quantile(d_rmse, 0.025) <= 0.0 <= np.quantile(d_rmse, 0.975)),
        "frac_molecules_improved": float(np.mean(np.abs(e_cand) < np.abs(e_base))),
    }


def evaluate_test(*, force: bool = False) -> dict[str, Any]:
    manifest = _frozen_manifest()
    if (C.EVAL / "metrics.json").exists() and not force:
        raise PermissionError("ESOL Test evaluation already exists; one-shot policy (use --force only for a bug fix)")
    views = D.build_views(include_test=True)
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    y_test = np.asarray(views["labels"]["y_test"], dtype=np.float64)
    rows_test = np.asarray(views["split"]["rows"]["test"], dtype=np.int64)
    rows_valid = np.asarray(views["split"]["rows"]["valid"], dtype=np.int64)
    C.EVAL.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    preds_valid: dict[str, np.ndarray] = {}
    preds_test: dict[str, np.ndarray] = {}
    for arm, frozen in sorted(manifest["arms"].items()):
        ckpt_path = Path(C.REPO / frozen["checkpoint"])
        if C.sha256_file(ckpt_path) != frozen["checkpoint_sha256"]:
            raise PermissionError(f"{arm}: checkpoint changed after the manifest freeze")
        model_kind = str(frozen["family"])
        seed = int(frozen["seed"])
        model, _meta = M.build(
            model_kind,
            views["spec"],
            views["train_norm"],
            seed,
            "cpu",
            dropout=float(frozen["dropout"]),
            physchem_norm=views["physchem_norm"],
        )
        M.load_checkpoint(model, ckpt_path, expected_sha256=frozen["checkpoint_sha256"], expected_family=model_kind, expected_seed=seed)
        physchem = _physchem(views, model_kind)
        pred_valid = T.denormalise(M.predict(model, views["valid_cache"], "cpu", physchem=physchem), views["target_norm"])
        if abs(float(C.all_metrics(pred_valid, y_valid)["rmse"]) - float(frozen["best_inner_valid_rmse"])) > TOL:
            raise RuntimeError(f"{arm}: inner Validation RMSE does not reproduce before the Test read")
        pred_test = T.denormalise(M.predict(model, views["test_cache"], "cpu", physchem=physchem), views["target_norm"])
        vm = C.all_metrics(pred_valid, y_valid)
        tm = C.all_metrics(pred_test, y_test)
        preds_valid[arm] = pred_valid
        preds_test[arm] = pred_test
        results[arm] = {
            "arm": arm,
            "model": frozen["model"],
            "family": model_kind,
            "seed": seed,
            "dropout": float(frozen["dropout"]),
            "checkpoint": frozen["checkpoint"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "best_epoch": int(frozen["best_epoch"]),
            "train_rmse": float(frozen["train_rmse"]),
            "train_mae": float(frozen["train_mae"]),
            "valid_rmse": float(vm["rmse"]),
            "valid_mae": float(vm["mae"]),
            "valid_r2": float(vm["r2"]),
            "test_rmse": float(tm["rmse"]),
            "test_mae": float(tm["mae"]),
            "test_r2": float(tm["r2"]),
            "parameters": int(frozen["parameters"]),
        }
        C.log(
            f"[test-eval] {arm} epoch={frozen['best_epoch']} valid={vm['rmse']:.6f} "
            f"test_rmse={tm['rmse']:.6f} test_mae={tm['mae']:.6f}"
        )

    arms = sorted(results)
    rows = [results[arm] for arm in arms]
    table_columns = [
        "model",
        "arm",
        "seed",
        "best_epoch",
        "train_rmse",
        "train_mae",
        "valid_rmse",
        "valid_mae",
        "valid_r2",
        "test_rmse",
        "test_mae",
        "test_r2",
    ]
    with open(C.EVAL / "table_runs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in table_columns} for row in rows)

    summary = _summarise(rows)
    C.write_json(C.EVAL / "summary.json", summary)
    with open(C.EVAL / "table_summary.csv", "w", newline="", encoding="utf-8") as handle:
        columns = [
            "model",
            "n_seeds",
            "train_rmse_mean",
            "train_rmse_sd",
            "valid_rmse_mean",
            "valid_rmse_sd",
            "valid_mae_mean",
            "valid_mae_sd",
            "test_rmse_mean",
            "test_rmse_sd",
            "test_mae_mean",
            "test_mae_sd",
            "test_r2_mean",
            "test_r2_sd",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model, rec in summary["models"].items():
            writer.writerow({"model": model, **{key: rec[key] for key in columns if key != "model"}})

    paired = _paired(arms, results, preds_test, y_test)
    C.write_json(C.EVAL / "paired.json", paired)
    if paired["pairs"]:
        with open(C.EVAL / "table_paired.csv", "w", newline="", encoding="utf-8") as handle:
            columns = [
                "pair",
                "seed",
                "delta_test_rmse",
                "delta_test_mae",
                "delta_valid_rmse",
                "delta_valid_mae",
                "frac_molecules_improved",
            ]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for rec in paired["per_seed"]:
                writer.writerow({key: rec[key] for key in columns})

    error_analysis = _error_analysis(views, rows, preds_test, rows_test, y_test)
    C.write_json(C.EVAL / "error_analysis.json", error_analysis)
    _write_per_molecule(views, rows, preds_test, preds_valid, rows_test, rows_valid, y_test, y_valid)

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "scope": "ESOL, one fixed Bemis-Murcko scaffold partition, one-shot Test read after manifest freeze",
        "split_manifest_sha256": C.sha256_file(C.SPLIT_MANIFEST),
        "manifest_sha256": manifest["manifest_sha256"],
        "test_labels_read_after_manifest_freeze": True,
        "test_used_for_selection": False,
        "units": "log(mol/L)",
        "models": results,
        "summary": summary,
        "paired": paired,
        "alignment": {
            "n_train": int(len(views["split"]["rows"]["train"])),
            "n_valid": int(rows_valid.size),
            "n_test": int(rows_test.size),
            "valid_rows_sha256": C.rows_sha256(rows_valid),
            "test_rows_sha256": C.rows_sha256(rows_test),
            "overlap_valid_test": int(len(set(rows_valid.tolist()) & set(rows_test.tolist()))),
            "y_test_sha256": C.vector_sha256_float64(y_test),
        },
    }
    C.write_json(C.EVAL / "metrics.json", payload)
    return payload


def _summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model"]), []).append(row)
    models: dict[str, Any] = {}
    for model, subset in by_model.items():
        subset = sorted(subset, key=lambda rec: int(rec["seed"]))
        rec: dict[str, Any] = {
            "n_seeds": len(subset),
            "seeds": [int(r["seed"]) for r in subset],
            "best_epochs": [int(r["best_epoch"]) for r in subset],
            "dropout": float(subset[0]["dropout"]),
        }
        for name, key in (
            ("train_rmse", "train_rmse"),
            ("train_mae", "train_mae"),
            ("valid_rmse", "valid_rmse"),
            ("valid_mae", "valid_mae"),
            ("test_rmse", "test_rmse"),
            ("test_mae", "test_mae"),
            ("test_r2", "test_r2"),
        ):
            mean, sd = _mean_std([float(r[key]) for r in subset])
            rec[f"{name}_mean"] = mean
            rec[f"{name}_sd"] = sd
        models[model] = rec
    return {"models": models}


def _paired(arms: Sequence[str], results: Mapping[str, Any], preds_test: Mapping[str, np.ndarray], y: np.ndarray) -> dict[str, Any]:
    base_arms = {int(results[a]["seed"]): a for a in arms if results[a]["model"] == C.MODEL_DISPLAY["base"]}
    per_seed: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for model in sorted({str(results[a]["model"]) for a in arms} - {C.MODEL_DISPLAY["base"]}):
        cand_arms = {int(results[a]["seed"]): a for a in arms if results[a]["model"] == model}
        for seed in sorted(set(base_arms) & set(cand_arms)):
            b = results[base_arms[seed]]
            c = results[cand_arms[seed]]
            boot = _paired_bootstrap(preds_test[base_arms[seed]], preds_test[cand_arms[seed]], y)
            per_seed.append(
                {
                    "pair": f"{model} - {C.MODEL_DISPLAY['base']}",
                    "model": model,
                    "seed": int(seed),
                    "delta_test_rmse": float(c["test_rmse"] - b["test_rmse"]),
                    "delta_test_mae": float(c["test_mae"] - b["test_mae"]),
                    "delta_valid_rmse": float(c["valid_rmse"] - b["valid_rmse"]),
                    "delta_valid_mae": float(c["valid_mae"] - b["valid_mae"]),
                    "frac_molecules_improved": boot["frac_molecules_improved"],
                    "bootstrap": boot,
                }
            )
        if any(rec["model"] == model for rec in per_seed):
            subset = [rec for rec in per_seed if rec["model"] == model]
            pairs.append(
                {
                    "pair": f"{model} - {C.MODEL_DISPLAY['base']}",
                    "n_seeds": len(subset),
                    "definition": "delta = candidate - BASE (negative = candidate better)",
                    "delta_test_rmse_mean": float(np.mean([rec["delta_test_rmse"] for rec in subset])),
                    "delta_test_rmse_sd": float(np.std([rec["delta_test_rmse"] for rec in subset], ddof=1)) if len(subset) > 1 else 0.0,
                    "delta_test_mae_mean": float(np.mean([rec["delta_test_mae"] for rec in subset])),
                    "delta_test_mae_sd": float(np.std([rec["delta_test_mae"] for rec in subset], ddof=1)) if len(subset) > 1 else 0.0,
                    "delta_valid_rmse_mean": float(np.mean([rec["delta_valid_rmse"] for rec in subset])),
                    "delta_valid_mae_mean": float(np.mean([rec["delta_valid_mae"] for rec in subset])),
                    "n_seeds_test_rmse_better": int(sum(1 for rec in subset if rec["delta_test_rmse"] < 0)),
                    "n_seeds_test_mae_better": int(sum(1 for rec in subset if rec["delta_test_mae"] < 0)),
                    "n_seeds_valid_rmse_better": int(sum(1 for rec in subset if rec["delta_valid_rmse"] < 0)),
                }
            )
    return {
        "definition": "delta = candidate - BASE; negative means the candidate is better",
        "per_seed": per_seed,
        "pairs": pairs,
        "bootstrap_note": (
            f"paired molecule bootstrap, {BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}; "
            "descriptive only, the split is a single partition"
        ),
    }


def _error_analysis(
    views: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    preds_test: Mapping[str, np.ndarray],
    rows_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    smiles = views["split"]["smiles"]
    split = S.load_split()
    scaffolds = split["scaffold_by_row"]
    names = list(C.PHYSCHEM_NAMES)
    phys = np.asarray(views["physchem"], dtype=np.float64)
    out: dict[str, Any] = {"models": {}}
    for row in rows:
        arm = str(row["arm"])
        pred = np.asarray(preds_test[arm], dtype=np.float64)
        err = pred - y_test
        abs_err = np.abs(err)
        order = np.argsort(-abs_err)
        top = [
            {
                "row_id": int(rows_test[i]),
                "smiles": str(smiles[int(rows_test[i])]),
                "scaffold": str(scaffolds[int(rows_test[i])]),
                "acyclic": bool(scaffolds[int(rows_test[i])] == ""),
                "y": float(y_test[i]),
                "pred": float(pred[i]),
                "err": float(err[i]),
                "abs_err": float(abs_err[i]),
                "n_atoms": int(views["test_cache"]["atom_ptr"][i + 1] - views["test_cache"]["atom_ptr"][i]),
            }
            for i in order[:10]
        ]
        # bias / RMSE by target-value bin (per-seed, descriptive)
        bins = np.linspace(float(y_test.min()), float(y_test.max()), 5)
        edges = np.unique(bins)
        bin_stats = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (y_test >= lo) & (y_test <= hi if hi == edges[-1] else y_test < hi)
            if not mask.any():
                continue
            bin_stats.append(
                {
                    "y_low": float(lo),
                    "y_high": float(hi),
                    "n": int(mask.sum()),
                    "mean_err": float(err[mask].mean()),
                    "rmse": float(np.sqrt(np.mean(err[mask] ** 2))),
                    "mae": float(np.mean(np.abs(err[mask]))),
                }
            )
        out["models"][arm] = {
            "model": row["model"],
            "seed": int(row["seed"]),
            "mean_signed_error": float(err.mean()),
            "median_abs_error": float(np.median(abs_err)),
            "p90_abs_error": float(np.quantile(abs_err, 0.90)),
            "max_abs_error": float(abs_err.max()),
            "sse_share_top_1pct": _sse_concentration(err, 0.01),
            "sse_share_top_5pct": _sse_concentration(err, 0.05),
            "sse_share_top_10pct": _sse_concentration(err, 0.10),
            "spearman_abs_err_vs_abs_y": _spearman(abs_err, np.abs(y_test)),
            "spearman_abs_err_vs_n_atoms": _spearman(
                abs_err, np.asarray([views["test_cache"]["atom_ptr"][i + 1] - views["test_cache"]["atom_ptr"][i] for i in range(y_test.size)], dtype=np.float64)
            ),
            "spearman_abs_err_vs_molwt": _spearman(abs_err, phys[rows_test][:, names.index("MolWt")]),
            "spearman_abs_err_vs_mollogp": _spearman(abs_err, phys[rows_test][:, names.index("MolLogP")]),
            "bins_by_target": bin_stats,
            "top10_abs_error_molecules": top,
        }
    return out


def _write_per_molecule(
    views: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    preds_test: Mapping[str, np.ndarray],
    preds_valid: Mapping[str, np.ndarray],
    rows_test: np.ndarray,
    rows_valid: np.ndarray,
    y_test: np.ndarray,
    y_valid: np.ndarray,
) -> None:
    smiles = views["split"]["smiles"]
    columns = ["row_id", "smiles", "y_logS"]
    for row in rows:
        columns += [f"pred_test_{row['arm']}", f"err_test_{row['arm']}", f"abs_err_test_{row['arm']}"]
    with open(C.EVAL / "per_molecule_test.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, row_id in enumerate(rows_test):
            record = {"row_id": int(row_id), "smiles": str(smiles[int(row_id)]), "y_logS": float(y_test[i])}
            for rec in rows:
                arm = str(rec["arm"])
                err = float(preds_test[arm][i] - y_test[i])
                record[f"pred_test_{arm}"] = float(preds_test[arm][i])
                record[f"err_test_{arm}"] = err
                record[f"abs_err_test_{arm}"] = abs(err)
            writer.writerow(record)
    columns = ["row_id", "smiles", "y_logS"]
    for row in rows:
        columns += [f"pred_valid_{row['arm']}", f"abs_err_valid_{row['arm']}"]
    with open(C.EVAL / "per_molecule_valid.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, row_id in enumerate(rows_valid):
            record = {"row_id": int(row_id), "smiles": str(smiles[int(row_id)]), "y_logS": float(y_valid[i])}
            for rec in rows:
                arm = str(rec["arm"])
                record[f"pred_valid_{arm}"] = float(preds_valid[arm][i])
                record[f"abs_err_valid_{arm}"] = abs(float(preds_valid[arm][i] - y_valid[i]))
            writer.writerow(record)


__all__ = ["TOL", "verify_arm", "freeze_manifest", "evaluate_test"]
