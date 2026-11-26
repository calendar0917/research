"""Analysis of the single `SCAFFOLD_BALANCED` run (PROTOCOL section 7).

Reads the run artefacts + the read-only historical FIXED1 predictions and writes
the main table, the full Valid curve, the per-molecule comparison, the paired
bootstrap, the trimmed sensitivity, the single training-support diagnostic, the
11 fixed cases, the optimisation diagnostics, the post-run checks, the
checkpoint manifest and the pre-registered verdict.

Nothing here changes the primary metric: best full official Valid RMSE, plain
per-molecule weighting.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import sb_common as C


RUN_DIR = C.RUNS_DIR / "scaffold_balanced_seed0"


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def load_new_run() -> dict[str, Any]:
    run = C.read_json(RUN_DIR / "run.json")
    if run.get("status") != "COMPLETED":
        raise RuntimeError(f"run status {run.get('status')} — analysis still runs but the verdict must be INCONCLUSIVE")
    curve = pd.read_csv(RUN_DIR / "train_curve.csv")
    best = np.load(RUN_DIR / "predictions_best.npz")
    final = np.load(RUN_DIR / "predictions_final.npz")
    last20 = np.load(RUN_DIR / "predictions_last20.npz")
    return {
        "run": run,
        "curve": curve,
        "best": {k: best[k] for k in best.files},
        "final": {k: final[k] for k in final.files},
        "last20": {k: last20[k] for k in last20.files},
    }


def load_historical() -> dict[str, Any]:
    run = C.read_json(C.CSC_RUN / "run.json")
    curve = pd.read_csv(C.CSC_RUN / "train_curve.csv")
    # only the Valid predictions of the historical best checkpoint are needed
    with np.load(C.CSC_RUN / "predictions_best.npz") as npz:
        best = {
            "valid_row_ids": npz["valid_row_ids"],
            "valid_pred_raw": npz["valid_pred_raw"],
            "valid_y_raw": npz["valid_y_raw"],
            "epoch": npz["epoch"],
        }
    with np.load(C.CSC_RUN / "predictions_last20.npz") as npz:
        last20 = {"valid_pred_raw": npz["valid_pred_raw"], "valid_y_raw": npz["valid_y_raw"]}
    with np.load(C.CSC_RUN / "predictions_final.npz") as npz:
        final = {"valid_pred_raw": npz["valid_pred_raw"], "valid_y_raw": npz["valid_y_raw"]}
    return {"run": run, "curve": curve, "best": best, "last20": last20, "final": final}


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    return {"rmse": C.rmse(p, t), "mae": C.mae(p, t), "mse": float(np.mean((p - t) ** 2))}


# ---------------------------------------------------------------------------
# Morgan support diagnostic (single, frozen)
# ---------------------------------------------------------------------------


def morgan_matrix(smiles: Sequence[str], *, radius: int, n_bits: int) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(n_bits))
    out = np.zeros((len(smiles), int(n_bits)), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"unparsable SMILES at {i}: {smi!r}")
        out[i] = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
    return out


def tanimoto_rows(query: np.ndarray, reference: np.ndarray, *, chunk: int = 256) -> np.ndarray:
    """Tanimoto of every query row against every reference row, chunked (memory safe)."""
    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    q_sum = q.sum(axis=1)
    r_sum = r.sum(axis=1)
    out = np.zeros((q.shape[0], r.shape[0]), dtype=np.float32)
    for start in range(0, q.shape[0], int(chunk)):
        block = q[start:start + int(chunk)]
        inter = block @ r.T
        denom = q_sum[start:start + int(chunk)][:, None] + r_sum[None, :] - inter
        out[start:start + int(chunk)] = inter / np.maximum(denom, 1e-12)
    return out


def support_diagnostic(train_smiles: Sequence[str], valid_smiles: Sequence[str]) -> dict[str, Any]:
    fp_train = morgan_matrix(train_smiles, radius=C.MORGAN_RADIUS, n_bits=C.MORGAN_BITS)
    fp_valid = morgan_matrix(valid_smiles, radius=C.MORGAN_RADIUS, n_bits=C.MORGAN_BITS)
    sim_vt = tanimoto_rows(fp_valid, fp_train)
    valid_max = sim_vt.max(axis=1)
    # Train leave-one-out max similarity, chunked
    loo = np.zeros(fp_train.shape[0], dtype=np.float32)
    for start in range(0, fp_train.shape[0], 512):
        block = fp_train[start:start + 512]
        sim = tanimoto_rows(block, fp_train)
        rows = np.arange(sim.shape[0])
        sim[rows, np.arange(start, start + sim.shape[0])] = -1.0
        loo[start:start + sim.shape[0]] = sim.max(axis=1)
        del sim
    threshold = float(np.median(loo))
    del sim_vt, fp_train, fp_valid
    return {
        "morgan": {"radius": C.MORGAN_RADIUS, "n_bits": C.MORGAN_BITS, "fingerprint": "Morgan count->bit, default RDKit generator"},
        "train_loo_max_tanimoto": C.percentile_summary(loo.astype(np.float64)),
        "threshold": threshold,
        "threshold_rule": "median of Train leave-one-out max Tanimoto (fixed, not searched)",
        "valid_max_tanimoto": C.percentile_summary(valid_max.astype(np.float64)),
        "valid_max_to_train": valid_max.astype(np.float64),
        "n_valid_above_threshold": int((valid_max >= threshold).sum()),
        "n_valid_below_threshold": int((valid_max < threshold).sum()),
    }


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------


def paired_analysis(
    pred_new: np.ndarray,
    pred_hist: np.ndarray,
    y: np.ndarray,
    *,
    row_ids: np.ndarray,
    support: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    p_new = np.asarray(pred_new, dtype=np.float64)
    p_hist = np.asarray(pred_hist, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    se_new = (p_new - t) ** 2
    se_hist = (p_hist - t) ** 2
    ae_new = np.abs(p_new - t)
    ae_hist = np.abs(p_hist - t)
    gain = se_hist - se_new           # > 0: SCAFFOLD_BALANCED is better
    n = int(t.size)
    rmse_new, rmse_hist = C.rmse(p_new, t), C.rmse(p_hist, t)
    mae_new, mae_hist = C.mae(p_new, t), C.mae(p_hist, t)

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(C.BOOTSTRAP_REPLICATES, n))
    boot_rmse_new = np.sqrt(se_new[idx].mean(axis=1))
    boot_rmse_hist = np.sqrt(se_hist[idx].mean(axis=1))
    boot_mae_new = ae_new[idx].mean(axis=1)
    boot_mae_hist = ae_hist[idx].mean(axis=1)
    d_rmse = boot_rmse_new - boot_rmse_hist
    d_mae = boot_mae_new - boot_mae_hist

    def _ci(a: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(a.mean()),
            "ci95_low": float(np.percentile(a, 2.5)),
            "ci95_high": float(np.percentile(a, 97.5)),
        }

    # trimmed sensitivity: drop the 5% largest *common* squared error
    common = np.minimum(se_new, se_hist)
    n_trim = int(round(C.TRIM_FRACTION * n))
    trim_idx = np.argsort(-common, kind="stable")[:n_trim]
    keep = np.ones(n, dtype=bool)
    keep[trim_idx] = False
    trimmed = {
        "n_dropped": int(n_trim),
        "rule": f"drop the {C.TRIM_FRACTION:.0%} of Valid molecules with the largest min(se_new, se_hist)",
        "n_remaining": int(keep.sum()),
        "rmse_new": C.rmse(p_new[keep], t[keep]),
        "rmse_hist": C.rmse(p_hist[keep], t[keep]),
        "delta_rmse": C.rmse(p_new[keep], t[keep]) - C.rmse(p_hist[keep], t[keep]),
        "mae_new": C.mae(p_new[keep], t[keep]),
        "mae_hist": C.mae(p_hist[keep], t[keep]),
        "delta_mae": C.mae(p_new[keep], t[keep]) - C.mae(p_hist[keep], t[keep]),
        "dropped_row_ids": [int(row_ids[i]) for i in trim_idx],
    }

    # concentration of the net squared-error gain
    order = np.argsort(-gain, kind="stable")
    top = order[:n_trim]
    net = float(gain.sum())
    pos = float(gain[gain > 0].sum())
    concentration = {
        "net_sse_change": net,
        "positive_part": pos,
        "negative_part": float(gain[gain < 0].sum()),
        "top_5pct_gain_sum": float(gain[top].sum()),
        "top_5pct_share_of_net_gain": float(gain[top].sum() / net) if net > 0 else None,
        "top_5pct_share_of_positive_part": float(gain[top].sum() / pos) if pos > 0 else None,
        "n_molecules": int(n),
        "top5_row_ids": [int(row_ids[i]) for i in top[:10]],
    }

    payload = {
        "n_molecules": n,
        "new": {"rmse": rmse_new, "mae": mae_new, "mse": float(np.mean(se_new))},
        "historical_fixed1": {"rmse": rmse_hist, "mae": mae_hist, "mse": float(np.mean(se_hist))},
        "delta": {
            "rmse": rmse_new - rmse_hist,
            "mae": mae_new - mae_hist,
            "mse": float(np.mean(se_new) - np.mean(se_hist)),
            "relative_rmse": (rmse_new - rmse_hist) / rmse_hist,
            "relative_mae": (mae_new - mae_hist) / mae_hist,
        },
        "per_molecule": {
            "improved_count": int((gain > 0).sum()),
            "degraded_count": int((gain < 0).sum()),
            "tied_count": int((gain == 0).sum()),
            "improved_fraction": float((gain > 0).mean()),
            "median_abs_error_new": float(np.median(ae_new)),
            "median_abs_error_hist": float(np.median(ae_hist)),
            "mean_gain": float(gain.mean()),
            "median_gain": float(np.median(gain)),
            "sum_positive_gain": pos,
            "sum_negative_gain": float(gain[gain < 0].sum()),
        },
        "bootstrap": {
            "replicates": int(C.BOOTSTRAP_REPLICATES),
            "seed": int(C.BOOTSTRAP_SEED),
            "delta_rmse": _ci(d_rmse),
            "delta_mae": _ci(d_mae),
            "fraction_replicates_delta_rmse_negative": float((d_rmse < 0).mean()),
            "fraction_replicates_delta_mae_negative": float((d_mae < 0).mean()),
            "unit": "paired molecule bootstrap over the 420 official Valid molecules",
        },
        "trimmed_sensitivity": trimmed,
        "concentration": concentration,
    }

    if support is not None:
        thr = float(support["threshold"])
        sims = np.asarray(support["valid_max_to_train"], dtype=np.float64)
        low = sims < thr
        high = ~low
        strata: dict[str, Any] = {
            "threshold": thr,
            "threshold_rule": support["threshold_rule"],
            "groups": {},
        }
        for name, mask in (("low_support", low), ("high_support", high)):
            if not mask.any():
                strata["groups"][name] = {"n": 0}
                continue
            m_new = metrics(p_new[mask], t[mask])
            m_hist = metrics(p_hist[mask], t[mask])
            strata["groups"][name] = {
                "n": int(mask.sum()),
                "row_ids": [int(x) for x in row_ids[mask]],
                "new": m_new,
                "historical_fixed1": m_hist,
                "delta_rmse": m_new["rmse"] - m_hist["rmse"],
                "delta_mae": m_new["mae"] - m_hist["mae"],
                "delta_mse": m_new["mse"] - m_hist["mse"],
                "improved_count": int((gain[mask] > 0).sum()),
                "degraded_count": int((gain[mask] < 0).sum()),
                "mean_max_tanimoto": float(sims[mask].mean()),
                "min_max_tanimoto": float(sims[mask].min()),
                "max_max_tanimoto": float(sims[mask].max()),
            }
        b_idx = np.random.default_rng(C.BOOTSTRAP_SEED + 1).integers(0, int(low.sum()), size=(C.BOOTSTRAP_REPLICATES, int(low.sum()))) if low.sum() else None
        if b_idx is not None:
            se_l_new = se_new[low][b_idx].mean(axis=1)
            se_l_hist = se_hist[low][b_idx].mean(axis=1)
            d = np.sqrt(se_l_new) - np.sqrt(se_l_hist)
            strata["groups"]["low_support"]["bootstrap_delta_rmse"] = _ci(d)
            strata["groups"]["low_support"]["fraction_replicates_negative"] = float((d < 0).mean())
        payload["support_strata"] = strata

    table = pd.DataFrame(
        {
            "row_id": np.asarray(row_ids, dtype=np.int64),
            "y_raw": t,
            "pred_scaffold_balanced": p_new,
            "pred_fixed1": p_hist,
            "error_scaffold_balanced": p_new - t,
            "error_fixed1": p_hist - t,
            "se_scaffold_balanced": se_new,
            "se_fixed1": se_hist,
            "delta_se": se_new - se_hist,
            "gain_positive_is_better": gain,
            "ae_scaffold_balanced": ae_new,
            "ae_fixed1": ae_hist,
            "max_train_tanimoto": (
                np.asarray(support["valid_max_to_train"], dtype=np.float64) if support is not None else np.nan
            ),
            "support_stratum": (
                np.where(np.asarray(support["valid_max_to_train"]) >= float(support["threshold"]), "high", "low")
                if support is not None
                else ""
            ),
        }
    )
    return payload, table


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    t0 = time.perf_counter()
    frozen = C.read_json(C.CONFIG_DIR / "experiment.json")
    checks = C.read_json(C.ANALYSIS_DIR / "checks.json")
    new = load_new_run()
    hist = load_historical()
    run = new["run"]
    curve = new["curve"]
    hcurve = hist["curve"]

    valid_ids_new = np.asarray(new["best"]["valid_row_ids"], dtype=np.int64)
    valid_ids_hist = np.asarray(hist["best"]["valid_row_ids"], dtype=np.int64)
    if not np.array_equal(valid_ids_new, valid_ids_hist):
        raise RuntimeError("Valid row ids differ between the new run and the historical run")
    y_valid = np.asarray(new["best"]["valid_y_raw"], dtype=np.float64)
    if np.max(np.abs(y_valid - np.asarray(hist["best"]["valid_y_raw"], dtype=np.float64))) > 1e-12:
        raise RuntimeError("Valid labels differ between the runs")
    p_new = np.asarray(new["best"]["valid_pred_raw"], dtype=np.float64)
    p_hist = np.asarray(hist["best"]["valid_pred_raw"], dtype=np.float64)

    # ---- official labels cross-check (Train/Valid only)
    mol = pd.read_csv(C.MOL_CSV).reset_index().rename(columns={"index": "row_id"})
    label_of = dict(zip(mol["row_id"].astype(int), mol["exp"].astype(float)))
    smiles_of = dict(zip(mol["row_id"].astype(int), mol["smiles"].astype(str)))
    official_y = np.asarray([label_of[int(r)] for r in valid_ids_new], dtype=np.float64)
    if np.max(np.abs(official_y - y_valid)) > 1e-6:
        raise RuntimeError("prediction file labels disagree with mapping/mol.csv.gz")

    # ---- support diagnostic (single, frozen)
    train_rows = pd.read_csv(C.SPLIT_DIR / "train.csv.gz", header=None)[0].to_numpy(dtype=np.int64)
    support = support_diagnostic([smiles_of[int(r)] for r in train_rows], [smiles_of[int(r)] for r in valid_ids_new])
    paired, per_mol = paired_analysis(
        p_new, p_hist, y_valid, row_ids=valid_ids_new, support=support
    )
    C.write_json(C.ANALYSIS_DIR / "support_strata.json", {k: v for k, v in support.items() if k != "valid_max_to_train"} | {"strata": paired.get("support_strata")})

    # ---- 11 fixed cases (descriptive)
    cases = pd.read_csv(C.SFL_CASES)
    case_rows = [int(x) for x in cases["row_id"].tolist()]
    case_df = per_mol[per_mol["row_id"].isin(case_rows)].copy()
    case_meta = cases.set_index("row_id")
    case_df["in_sets"] = case_df["row_id"].map(lambda r: str(case_meta.loc[r, "in_sets"]))
    case_df["case_figure"] = case_df["row_id"].map(lambda r: str(case_meta.loc[r, "figure"]))
    cols = [
        "row_id", "in_sets", "y_raw", "pred_fixed1", "pred_scaffold_balanced",
        "error_fixed1", "error_scaffold_balanced", "se_fixed1", "se_scaffold_balanced",
        "delta_se", "max_train_tanimoto", "support_stratum", "case_figure",
    ]
    case_df = case_df[cols].sort_values("row_id")
    C.write_csv(C.ANALYSIS_DIR / "fixed_cases.csv", case_df.to_dict("records"), cols)
    fixed_cases = {
        "source": C.rel(C.SFL_CASES),
        "source_sha256": C.sha256_file(C.SFL_CASES),
        "n_cases": int(len(case_rows)),
        "cases": case_df.to_dict("records"),
        "summary": {
            "n_improved": int((case_df["delta_se"] < 0).sum()),
            "n_degraded": int((case_df["delta_se"] > 0).sum()),
            "mean_abs_prediction_change": float(
                np.mean(np.abs(case_df["pred_scaffold_balanced"] - case_df["pred_fixed1"]))
            ),
            "rmse_fixed1_on_cases": C.rmse(case_df["pred_fixed1"].to_numpy(), case_df["y_raw"].to_numpy()),
            "rmse_new_on_cases": C.rmse(case_df["pred_scaffold_balanced"].to_numpy(), case_df["y_raw"].to_numpy()),
        },
    }
    C.write_json(C.ANALYSIS_DIR / "fixed_cases.json", fixed_cases)

    # ---- per-molecule table (plus the whole Valid prediction change)
    per_mol["abs_prediction_change"] = np.abs(per_mol["pred_scaffold_balanced"] - per_mol["pred_fixed1"])
    per_mol.to_csv(C.ANALYSIS_DIR / "per_molecule_valid.csv", index=False)

    # ---- last-20 / final descriptive comparisons
    def _cmp(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
        m_new = metrics(a["valid_pred_raw"], y_valid)
        m_hist = metrics(b["valid_pred_raw"], y_valid)
        return {
            "new": m_new,
            "historical_fixed1": m_hist,
            "delta_rmse": m_new["rmse"] - m_hist["rmse"],
            "delta_mae": m_new["mae"] - m_hist["mae"],
        }

    last20_cmp = _cmp(new["last20"], hist["last20"])
    final_cmp = _cmp(new["final"], hist["final"])

    # ---- main table
    def _row(name: str, run_like: Mapping[str, Any], *, kind: str, params: int, epoch: int | None,
             best: Mapping[str, Any], final: Mapping[str, Any], last20: Mapping[str, Any],
             seconds: float | None, gpu_hours: float, test_scored: bool,
             extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        out = {
            "round": C.ROUND_ID,
            "dataset": C.DATASET,
            "arm": name,
            "kind": kind,
            "parameters": int(params),
            "best_epoch": epoch,
            "best_train_rmse_raw": best.get("train_rmse_raw"),
            "best_train_mae_raw": best.get("train_mae_raw"),
            "best_valid_rmse_raw": best["valid_rmse_raw"],
            "best_valid_mae_raw": best["valid_mae_raw"],
            "final_epoch": final.get("epoch"),
            "final_train_rmse_raw": final.get("train_rmse_raw"),
            "final_valid_rmse_raw": final.get("valid_rmse_raw"),
            "final_valid_mae_raw": final.get("valid_mae_raw"),
            "last20_valid_rmse_raw": last20.get("valid_rmse_raw"),
            "last20_valid_mae_raw": last20.get("valid_mae_raw"),
            "wall_seconds": seconds,
            "gpu_hours": float(gpu_hours),
            "test_scored": bool(test_scored),
        }
        if extra:
            out.update(extra)
        return out

    h_run = hist["run"]
    main_rows = [
        _row(
            "SCAFFOLD_BALANCED",
            run,
            kind="this_round",
            params=int(run["parameters"]),
            epoch=int(run["best_epoch"]),
            best={
                "train_rmse_raw": run["best_metrics"]["train"]["rmse_raw"],
                "train_mae_raw": run["best_metrics"]["train"]["mae_raw"],
                "valid_rmse_raw": run["best_metrics"]["valid"]["rmse_raw"],
                "valid_mae_raw": run["best_metrics"]["valid"]["mae_raw"],
            },
            final={"epoch": run["final_metrics"]["epoch"], **{k: v for k, v in run["final_metrics"].items() if k.startswith(("valid_", "train_"))}},
            last20={k: v for k, v in run["last20_mean_metrics"].items() if k.startswith("valid_")},
            seconds=float(run["total_seconds"]),
            gpu_hours=0.0,
            test_scored=False,
            extra={"loss": "weighted MSE", "device": run.get("device", "cpu")},
        ),
        _row(
            "FIXED1",
            h_run,
            kind="read_only_historical",
            params=int(h_run["parameters"]),
            epoch=int(h_run["best_epoch"]),
            best={
                "train_rmse_raw": h_run["best_metrics"]["train"]["rmse_raw"],
                "train_mae_raw": h_run["best_metrics"]["train"]["mae_raw"],
                "valid_rmse_raw": h_run["best_metrics"]["valid"]["rmse_raw"],
                "valid_mae_raw": h_run["best_metrics"]["valid"]["mae_raw"],
            },
            final={"epoch": h_run["final_metrics"]["epoch"], **{k: v for k, v in h_run["final_metrics"].items() if k.startswith(("valid_", "train_"))}},
            last20={k: v for k, v in h_run["last20_mean_metrics"].items() if k.startswith("valid_")},
            seconds=float(h_run["total_seconds"]),
            gpu_hours=float(h_run.get("gpu_hours", 0.0)),
            test_scored=bool(h_run.get("test_scored_every_epoch", False)),
            extra={"loss": "plain MSE", "device": h_run.get("environment", {}).get("device_name")},
        ),
    ]
    C.write_csv(
        C.ANALYSIS_DIR / "main_table.csv",
        main_rows,
        list(main_rows[0].keys()),
    )
    C.write_json(
        C.ANALYSIS_DIR / "historical_reference.json",
        {
            "fixed1": {
                "source": C.rel(C.CSC_RUN / "run.json"),
                "sha256": C.sha256_file(C.CSC_RUN / "run.json"),
                "predictions_best_sha256": C.sha256_file(C.CSC_RUN / "predictions_best.npz"),
                "curve_sha256": C.sha256_file(C.CSC_RUN / "train_curve.csv"),
                "values": C.HIST,
                "recomputed_valid_rmse_from_predictions": C.rmse(p_hist, y_valid),
                "recomputed_valid_mae_from_predictions": C.mae(p_hist, y_valid),
                "delta_vs_recorded": abs(C.rmse(p_hist, y_valid) - C.HIST["best_valid_rmse_raw"]),
            },
            "C1": {"source": C.rel(C.C1_MAIN_TABLE), "sha256": C.sha256_file(C.C1_MAIN_TABLE), "values": C.C1_REF},
            "GINE": {"source": "lcrc_encoder_boundary_reference_v0/analysis/main_table.csv", "values": C.GINE_REF},
            "ebr_paired_errors": {"source": C.rel(C.EBR_PAIRED), "sha256": C.sha256_file(C.EBR_PAIRED)},
        },
    )

    # ---- full Valid curve, epoch aligned
    curve_out = curve[["epoch", "train_rmse_raw", "train_mae_raw", "valid_rmse_raw", "valid_mae_raw",
                       "train_weighted_mse_trainmode", "train_mse_unweighted_trainmode",
                       "train_loss_step_mean", "learning_rate", "grad_norm_mean", "grad_norm_max",
                       "clip_events", "epoch_seconds", "cumulative_seconds", "is_best"]].copy()
    curve_out = curve_out.rename(columns={
        "train_rmse_raw": "train_rmse_raw_new", "train_mae_raw": "train_mae_raw_new",
        "valid_rmse_raw": "valid_rmse_raw_new", "valid_mae_raw": "valid_mae_raw_new",
    })
    h = hcurve[["epoch", "train_rmse_raw", "train_mae_raw", "valid_rmse_raw", "valid_mae_raw"]].rename(columns={
        "train_rmse_raw": "train_rmse_raw_fixed1", "train_mae_raw": "train_mae_raw_fixed1",
        "valid_rmse_raw": "valid_rmse_raw_fixed1", "valid_mae_raw": "valid_mae_raw_fixed1",
    })
    curve_out = curve_out.merge(h, on="epoch", how="left", validate="one_to_one")
    curve_out.to_csv(C.ANALYSIS_DIR / "valid_curve.csv", index=False)

    # same-epoch Train comparison
    best_ep = int(run["best_epoch"])
    hist_best_ep = int(h_run["best_epoch"])
    row_new_best = curve[curve.epoch == best_ep].iloc[0]
    row_new_at_hist = curve[curve.epoch == hist_best_ep].iloc[0]
    row_hist_at_new_best = hcurve[hcurve.epoch == best_ep].iloc[0]
    row_hist_best = hcurve[hcurve.epoch == hist_best_ep].iloc[0]
    epoch_aligned = {
        "new_best_epoch": best_ep,
        "historical_best_epoch": hist_best_ep,
        "at_new_best_epoch": {
            "epoch": best_ep,
            "new_train_rmse_raw": float(row_new_best["train_rmse_raw"]),
            "new_train_mae_raw": float(row_new_best["train_mae_raw"]),
            "fixed1_train_rmse_raw": float(row_hist_at_new_best["train_rmse_raw"]),
            "fixed1_train_mae_raw": float(row_hist_at_new_best["train_mae_raw"]),
            "new_valid_rmse_raw": float(row_new_best["valid_rmse_raw"]),
            "fixed1_valid_rmse_raw": float(row_hist_at_new_best["valid_rmse_raw"]),
            "train_rmse_delta": float(row_new_best["train_rmse_raw"] - row_hist_at_new_best["train_rmse_raw"]),
        },
        "at_historical_best_epoch": {
            "epoch": hist_best_ep,
            "new_train_rmse_raw": float(row_new_at_hist["train_rmse_raw"]),
            "new_train_mae_raw": float(row_new_at_hist["train_mae_raw"]),
            "fixed1_train_rmse_raw": float(row_hist_best["train_rmse_raw"]),
            "fixed1_train_mae_raw": float(row_hist_best["train_mae_raw"]),
            "new_valid_rmse_raw": float(row_new_at_hist["valid_rmse_raw"]),
            "fixed1_valid_rmse_raw": float(row_hist_best["valid_rmse_raw"]),
            "train_rmse_delta": float(row_new_at_hist["train_rmse_raw"] - row_hist_best["train_rmse_raw"]),
        },
        "note": "train_rmse_raw is the plain per-molecule Train metric (unweighted) at that epoch in both runs",
    }
    C.write_json(C.ANALYSIS_DIR / "epoch_aligned_train.json", epoch_aligned)

    # ---- curve statistics (best-epoch selection is a single-epoch order statistic)
    def _curve_stats(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for label, lo, hi in (("all_epochs", 1, C.EPOCHS), ("epochs_100_360", 100, C.EPOCHS), ("last_20", C.EPOCHS - 19, C.EPOCHS)):
            v = frame.loc[(frame["epoch"] >= lo) & (frame["epoch"] <= hi), "valid_rmse_raw"].to_numpy(dtype=np.float64)
            if v.size == 0:
                continue
            out[label] = {
                "n_epochs": int(v.size),
                "mean": float(v.mean()),
                "std": float(v.std(ddof=0)),
                "median": float(np.median(v)),
                "min": float(v.min()),
                "max": float(v.max()),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
            }
        out["n_epochs_valid_rmse_below_0.73"] = int((frame["valid_rmse_raw"] < 0.73).sum())
        out["n_epochs_valid_rmse_below_0.7274"] = int((frame["valid_rmse_raw"] < 0.7274).sum())
        return {"prefix": prefix, **out}

    curve_stats = {
        "note": (
            "the selected best epoch is a single-epoch order statistic of a fluctuating curve; "
            "the epoch-to-epoch spread is reported so a 0.0007 difference is not over-read"
        ),
        "new": _curve_stats(curve, "scaffold_balanced"),
        "fixed1": _curve_stats(hcurve, "fixed1"),
        "correlation_between_valid_curves_same_epoch": float(
            np.corrcoef(
                curve["valid_rmse_raw"].to_numpy(dtype=np.float64),
                hcurve["valid_rmse_raw"].to_numpy(dtype=np.float64),
            )[0, 1]
        ),
    }
    C.write_json(C.ANALYSIS_DIR / "curve_statistics.json", curve_stats)

    # ---- weight conventions (the prompt's "scaffold total weight" has two normalisations)
    # keep_default_na=False: the empty (acyclic) scaffold is the string "" and must stay a group
    sw = pd.read_csv(C.ANALYSIS_DIR / "scaffold_weights.csv", keep_default_na=False)
    grp = sw.groupby("scaffold")["weight"].agg(sum_weight="sum", n="count")
    n_tr = int(len(sw))
    k_tr = int(frozen["weights"]["k_train"])
    nv = np.asarray(grp["n"], dtype=np.float64)
    weight_conventions = {
        "note": (
            "'scaffold total weight' is reported in two normalisations: (a) the sum of the per-molecule "
            "weights that enter the gradient; (b) the same sum divided by N_train (a probability-like share). "
            "The max/min ratio is identical in both."
        ),
        "n_train": n_tr,
        "k_train": k_tr,
        "per_molecule_weight": {
            "plain_objective": 1.0,
            "balanced_objective": C.percentile_summary(np.asarray(sw["weight"], dtype=np.float64)),
        },
        "scaffold_total_in_weight_units": {
            "plain_objective": {"min": float(nv.min()), "max": float(nv.max()), "max_over_min": float(nv.max() / nv.min())},
            "balanced_objective": {
                "min": float(grp["sum_weight"].min()),
                "max": float(grp["sum_weight"].max()),
                "value": float(n_tr / k_tr),
                "max_over_min": float(grp["sum_weight"].max() / grp["sum_weight"].min()),
            },
        },
        "scaffold_total_normalized_by_N_train": {
            "plain_objective": {"min": float(nv.min() / n_tr), "max": float(nv.max() / n_tr), "max_over_min": float(nv.max() / nv.min())},
            "balanced_objective": {
                "value": float(1.0 / k_tr),
                "max_over_min": float(grp["sum_weight"].max() / grp["sum_weight"].min()),
            },
        },
        "n_scaffolds_with_total_weight_changed": int(k_tr),
        "molecules_with_weight_above_one": int((sw["weight"] > 1.0).sum()),
        "molecules_with_weight_below_one": int((sw["weight"] < 1.0).sum()),
    }
    C.write_json(C.ANALYSIS_DIR / "weight_conventions.json", weight_conventions)

    # ---- optimisation diagnostics
    clip_total = float(curve["clip_events"].sum())
    steps_total = float(curve["n_steps"].sum()) if "n_steps" in curve else float(len(curve) * C.steps_per_epoch(C.DATASET))
    diagnostics = {
        "epochs_run": int(len(curve)),
        "steps_per_epoch": int(C.steps_per_epoch(C.DATASET)),
        "total_steps": int(steps_total),
        "grad_clip_threshold": float(C.RECIPE["gradient_clip"]),
        "clip_events_total": int(clip_total),
        "clip_event_fraction": float(clip_total / steps_total),
        "epochs_with_any_clip": int((curve["clip_events"] > 0).sum()),
        "max_clip_events_in_an_epoch": int(curve["clip_events"].max()),
        "historical_fixed1": {
            "clip_events_total": int(hcurve["clip_events"].sum()),
            "clip_event_fraction": float(hcurve["clip_events"].sum() / (C.steps_per_epoch(C.DATASET) * len(hcurve))),
            "epochs_with_any_clip": int((hcurve["clip_events"] > 0).sum()),
            "max_clip_events_in_an_epoch": int(hcurve["clip_events"].max()),
            "grad_norm_mean_over_epochs": float(hcurve["grad_norm_mean"].mean()),
            "grad_norm_max": float(hcurve["grad_norm_max"].max()),
            "first_5_epoch_clip_events": int(hcurve["clip_events"].head(5).sum()),
            "last_20_epoch_clip_events": int(hcurve["clip_events"].tail(20).sum()),
        },
        "same_regime_note": (
            "clipping is an inherited property of the frozen recipe (LR 3e-4, batch 128, clip 5): "
            "the historical plain-MSE FIXED1 run clipped in 43.3% of its steps, this weighted run in 47.7%. "
            "The weighted arm is therefore not in an anomalous optimiser regime."
        ),
        "clip_events_first_5_epochs": int(curve["clip_events"].head(5).sum()),
        "clip_events_last_20_epochs": int(curve["clip_events"].tail(20).sum()),
        "grad_norm_mean_over_epochs": float(curve["grad_norm_mean"].mean()),
        "grad_norm_max": float(curve["grad_norm_max"].max()),
        "final_epoch_grad_norm_mean": float(curve["grad_norm_mean"].iloc[-1]),
        "lr_first_step": float(curve["learning_rate"].iloc[0]),
        "lr_last_step": float(curve["learning_rate"].iloc[-1]),
        "lr_schedule_expected_first": float(C.lr_at(0, C.total_steps(C.EPOCHS))),
        "lr_schedule_expected_last": float(C.lr_at(C.total_steps(C.EPOCHS) - 1, C.total_steps(C.EPOCHS))),
        "weighted_loss_first_epoch": float(curve["train_loss_step_mean"].iloc[0]),
        "weighted_loss_last_epoch": float(curve["train_loss_step_mean"].iloc[-1]),
        "unweighted_mse_first_epoch": float(curve["train_mse_unweighted_trainmode"].iloc[0]),
        "unweighted_mse_last_epoch": float(curve["train_mse_unweighted_trainmode"].iloc[-1]),
        "final_train_rmse_raw": float(curve["train_rmse_raw"].iloc[-1]),
        "final_valid_rmse_raw": float(curve["valid_rmse_raw"].iloc[-1]),
        "best_epoch": best_ep,
        "best_valid_rmse_raw": float(run["best_valid_rmse_raw"]),
        "monotone_decrease_of_weighted_train_loss": bool(
            float(curve["train_loss_step_mean"].iloc[-1]) < float(curve["train_loss_step_mean"].iloc[0])
        ),
        "nan_or_inf_in_curve": bool(
            not np.all(np.isfinite(curve[["train_rmse_raw", "valid_rmse_raw", "train_loss_step_mean"]].to_numpy()))
        ),
        "train_seconds": float(run["train_seconds"]),
        "eval_seconds": float(run["eval_seconds"]),
        "total_seconds": float(run["total_seconds"]),
        "wall_minutes": float(run["total_seconds"] / 60.0),
        "gpu_hours": 0.0,
        "gpu_hours_hard_cap": 0.25,
        "wall_hard_minutes": 40,
        "seconds_per_epoch_mean": float(curve["epoch_seconds"].mean()),
        "note": (
            "weighted train MSE is lower than the unweighted Train MSE by construction: the weights put "
            "up to N/K on singleton scaffolds. Both are reported so a fit change cannot be mistaken for an "
            "optimisation failure."
        ),
    }
    C.write_json(C.ANALYSIS_DIR / "training_diagnostics.json", diagnostics)

    # ---- post-run checks
    post = {
        "predictions_reproduce_run_best_valid_rmse": bool(
            abs(C.rmse(p_new, y_valid) - float(run["best_valid_rmse_raw"])) < 1e-12
        ),
        "recomputed_best_valid_rmse": C.rmse(p_new, y_valid),
        "run_best_valid_rmse": float(run["best_valid_rmse_raw"]),
        "predictions_reproduce_run_best_valid_mae": bool(
            abs(C.mae(p_new, y_valid) - float(run["best_metrics"]["valid"]["mae_raw"])) < 1e-12
        ),
        "last20_predictions_match_curve": bool(
            abs(
                C.rmse(np.asarray(new["last20"]["valid_pred_raw"], dtype=np.float64), y_valid)
                - float(np.mean(curve["valid_rmse_raw"].iloc[-20:]))
            )
            < 1e-3
        ),
        "final_epoch_is_360": bool(int(run["final_metrics"]["epoch"]) == C.EPOCHS),
        "global_steps_is_9720": bool(int(run["global_steps_run"]) == C.total_steps(C.EPOCHS)),
        "status_completed": bool(run["status"] == "COMPLETED"),
        "checks_passed": bool(checks["all_passed"]),
        "test_scored": bool(run.get("test_scored", False)),
        "splits_evaluated": ["train", "valid"],
        "curve_epochs_monotone": bool(np.all(np.diff(curve["global_step"].to_numpy()) > 0)),
        "weight_mean_is_one": bool(abs(float(frozen["weights"]["audit"]["mean_weight"]) - 1.0) <= 1e-12),
        "per_scaffold_equal_total": bool(frozen["weights"]["audit"]["all_per_scaffold_totals_equal"]),
        "weight_vector_sha256_matches_frozen": bool(run["weight_sha256"] == frozen["weights"]["sha256"]),
        "plan_matches_historical": bool(run["train_plan_matches_historical"]),
        "initial_state_matches_historical": bool(checks["checks"]["1_initial_state"]["initial_matches_historical"]),
    }
    post["all_ok"] = all(bool(v) for k, v in post.items() if k.startswith(("predictions_", "last20_", "final_", "global_", "status_", "checks_", "curve_", "weight_", "per_", "plan_", "initial_")) and not k.startswith("test_"))
    C.write_json(C.ANALYSIS_DIR / "post_checks.json", post)

    # ---- paired comparisons payload
    paired_payload = {
        "created_utc": C.utcnow(),
        "primary_metric": "valid_rmse_raw at the best epoch (full official Valid, plain per-molecule weighting)",
        "best_checkpoint_comparison": paired,
        "last20_comparison": last20_cmp,
        "final_epoch_comparison": final_cmp,
        "epoch_aligned_train": epoch_aligned,
        "prediction_change": {
            "rms_change": float(np.sqrt(np.mean((p_new - p_hist) ** 2))),
            "mae_change": float(np.mean(np.abs(p_new - p_hist))),
            "max_abs_change": float(np.max(np.abs(p_new - p_hist))),
            "correlation": float(np.corrcoef(p_new, p_hist)[0, 1]),
            "mean_signed_change": float(np.mean(p_new - p_hist)),
        },
    }
    C.write_json(C.ANALYSIS_DIR / "paired_comparisons.json", paired_payload)

    # ---- support strata csv
    strata = paired["support_strata"]
    strata_rows = []
    for name, g in strata["groups"].items():
        strata_rows.append({
            "group": name,
            "n": g["n"],
            "rmse_new": g["new"]["rmse"],
            "rmse_fixed1": g["historical_fixed1"]["rmse"],
            "delta_rmse": g["delta_rmse"],
            "mae_new": g["new"]["mae"],
            "mae_fixed1": g["historical_fixed1"]["mae"],
            "delta_mae": g["delta_mae"],
            "delta_mse": g["delta_mse"],
            "improved_count": g["improved_count"],
            "degraded_count": g["degraded_count"],
            "mean_max_tanimoto": g["mean_max_tanimoto"],
            "min_max_tanimoto": g["min_max_tanimoto"],
            "max_max_tanimoto": g["max_max_tanimoto"],
        })
    C.write_csv(C.ANALYSIS_DIR / "support_strata.csv", strata_rows, list(strata_rows[0].keys()))

    # ---- verdict (pre-registered)
    d_best = float(run["best_valid_rmse_raw"]) - C.HIST["best_valid_rmse_raw"]
    d_mae = float(run["best_metrics"]["valid"]["mae_raw"]) - C.HIST["best_valid_mae_raw"]
    d_endpoint = float(run["final_metrics"]["valid_rmse_raw"]) - C.HIST["endpoint_valid_rmse_raw"]
    d_last20 = float(run["last20_mean_metrics"]["valid_rmse_raw"]) - C.HIST["last20_valid_rmse_raw"]
    promising = bool(
        d_best <= C.VERDICT["rmse_delta_gain"]
        and d_endpoint < 0
        and d_last20 < 0
        and d_mae <= C.VERDICT["mae_no_degradation"]
    )
    tail_share = paired["concentration"]["top_5pct_share_of_net_gain"]
    tail_only = bool(d_best < 0 and (tail_share is not None and tail_share >= 0.75))
    if not post["all_ok"] or run["status"] != "COMPLETED":
        label = "INCONCLUSIVE"
    elif promising:
        label = "STRUCTURAL_DISTRIBUTION_PROMISING"
    elif d_best < 0 or tail_only:
        label = "WEAK_OR_TAIL_ONLY_GAIN"
    else:
        label = "NO_CLEAR_GAIN"
    verdict = {
        "created_utc": C.utcnow(),
        "round_id": C.ROUND_ID,
        "label": label,
        "primary_metric": "best full official Valid RMSE (420 molecules, plain per-molecule weighting)",
        "inputs": {
            "best_valid_rmse_new": float(run["best_valid_rmse_raw"]),
            "best_valid_rmse_fixed1": C.HIST["best_valid_rmse_raw"],
            "delta_best_rmse": d_best,
            "delta_best_mae": d_mae,
            "delta_endpoint_rmse": d_endpoint,
            "delta_last20_rmse": d_last20,
            "threshold_delta_best": C.VERDICT["rmse_delta_gain"],
            "threshold_mae_degradation": C.VERDICT["mae_no_degradation"],
            "top_5pct_share_of_net_gain": tail_share,
            "tail_only_rule": "delta_best < 0 and top-5% molecules carry >= 75% of the net squared-error gain",
        },
        "rules": {
            "STRUCTURAL_DISTRIBUTION_PROMISING": "delta_best <= -0.03 AND endpoint < 0 AND last20 < 0 AND delta_mae <= +0.005",
            "WEAK_OR_TAIL_ONLY_GAIN": "small gain (delta_best < 0 without the PROMISING rule) or gain concentrated in a few cases",
            "NO_CLEAR_GAIN": "no improvement or clear degradation",
            "INCONCLUSIVE": "data / implementation / budget anomaly (failed checks or incomplete run)",
        },
        "beats_historical_C1": bool(float(run["best_valid_rmse_raw"]) < C.C1_REF["valid_rmse_raw"]),
        "C1_valid_rmse": C.C1_REF["valid_rmse_raw"],
        "beats_gine_reference": bool(float(run["best_valid_rmse_raw"]) < C.GINE_REF["valid_rmse_raw"]),
        "promising_rule_satisfied": promising,
        "tail_only_flag": tail_only,
        "low_support_group_improved": bool(paired["support_strata"]["groups"]["low_support"]["delta_rmse"] < 0)
        if paired["support_strata"]["groups"].get("low_support", {}).get("n", 0) > 0 else None,
        "caveats": [
            "single seed; the official Valid split has been used many times in this repository",
            "0.03 is a resource-allocation reference, not a significance threshold",
            "the support split is a descriptive diagnostic; it does not replace the official full-Valid metric",
            "no Test data was read, scored or used",
        ],
    }
    C.write_json(C.ANALYSIS_DIR / "verdict.json", verdict)

    # ---- checkpoint manifest
    files = {}
    for name in ("checkpoint_best.pt", "checkpoint_final.pt", "predictions_best.npz",
                 "predictions_final.npz", "predictions_last20.npz", "train_curve.csv",
                 "training_order.json", "best_checkpoint_eval.json", "run.json"):
        path = RUN_DIR / name
        if path.is_file():
            files[name] = {"path": C.rel(path), "bytes": int(path.stat().st_size), "sha256": C.sha256_file(path)}
    manifest = {
        "created_utc": C.utcnow(),
        "round_id": C.ROUND_ID,
        "run_dir": C.rel(RUN_DIR),
        "files": files,
        "checkpoint_selection": "minimum full official Valid RMSE over 360 epochs (strictly smaller replaces)",
        "best_epoch": int(run["best_epoch"]),
        "best_valid_rmse_raw": float(run["best_valid_rmse_raw"]),
        "parameters": int(run["parameters"]),
        "initial_state_sha256": run["model_audit"]["initial_state_sha256"],
        "historical_initial_state_sha256": C.HIST["initial_state_sha256"],
        "weight_sha256": run["weight_sha256"],
        "train_plan_sha256": run["train_plan_sha256"],
        "test_scored": False,
    }
    C.write_json(C.RESULTS_DIR / "manifest.json", manifest)

    C.log(f"analysis done in {time.perf_counter() - t0:.1f}s verdict={label} delta_best={d_best:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
