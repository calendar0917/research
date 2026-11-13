"""Part-B forward-only evaluation of one trained task against its own oracle.

Main CF scope is ``FULL`` (the whole new molecule is re-encoded; ``G_FIXED`` is
only an internal diagnostic and is not used here).  For every frozen
Valid/Test main pair and view ``V in {T, K0}``:

``delta_pred = f(V) - f(O)``, ``delta_ref = oracle(V) - oracle(O)``.

The WC_LOGP primary quantity is the per-family mean ``|delta_pred - delta_ref|``
against the fixed zero-change baseline ``mean |delta_ref|`` (``gain = baseline -
model``).  The MOLWT reference change is exactly 0, so the report is the absolute
model movement ``|delta_pred|`` and its scale, never a signed mean.

Uncertainty: one shared scaffold-cluster bootstrap per split (5000 replicates,
seed 20261114), molecule-equal / instance-equal / control-equal weights, the same
resampling used for both tasks and all baselines/families of that split.

This stage performs no optimisation; it forwards the already selected best
checkpoint only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import round_common as RC
import train_task as TT

PRIMARY_SCOPE = RC.CF_SCOPE


def TT_labels(task: str, split: str, cache) -> np.ndarray:
    lab = TT.load_task_labels(task, split)
    rows = [int(x) for x in np.asarray(cache["rows"]).tolist()]
    return np.asarray([lab[r] for r in rows], dtype=np.float64)


def raw_metrics(pred_std: np.ndarray, y_raw: np.ndarray, mean: float, scale: float) -> dict[str, float]:
    return TT.raw_metrics(pred_std, y_raw, mean, scale)


def _f(x: Any) -> float:
    try:
        s = str(x).strip()
        return float(s) if s not in ("", "nan", "None") else float("nan")
    except Exception:
        return float("nan")


def _predict_cache(model, cache, device, batch_size: int = RC.EVAL_BATCH_SIZE) -> np.ndarray:
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            idx = list(range(start, min(start + batch_size, n)))
            batch = RC.SC.DD.PREP_DATA.collate(cache, idx)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out.append(model(batch).detach().float().cpu().numpy().reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


def build_split_graph_cache(split: str, spec) -> tuple[dict[str, Any], list[Mapping[str, str]]]:
    graph_rows = RC.read_csv(RC.SOURCE_ANALYSIS / "graph_manifest.csv")
    raw_rows = RC.SC.load_raw_rows()["rows"]["Lipophilicity"]
    entries, kept = RC.INF.build_split_cache(RC.SOURCE_ANALYSIS, graph_rows, split, raw_rows, spec)
    cache = RC.FEAT.build_cache(entries, spec, split=f"oracle_cf_{split}")
    return cache, kept, graph_rows


def _verify_cache(cache, kept) -> dict[str, Any]:
    atom_counts = np.diff(np.asarray(cache["atom_ptr"]))
    bond_counts = np.diff(np.asarray(cache["bond_ptr"]))
    comp = np.asarray(cache["n_components"])
    idx = {str(g): i for i, g in enumerate(cache["view_ids"])}
    o_atoms = {str(r["graph_id"]): int(atom_counts[idx[str(r["graph_id"])]]) for r in kept if str(r["kind"]) == "O"}
    o_bonds = {str(r["graph_id"]): int(bond_counts[idx[str(r["graph_id"])]]) for r in kept if str(r["kind"]) == "O"}
    bad_comp = int((comp != 1).sum())
    bad_atoms = 0
    bad_bonds = 0
    for r in kept:
        gid = str(r["graph_id"])
        rid = int(r["row_id"])
        o_gid = f"{rid}:G:[]+[]"
        if gid == o_gid:
            continue
        if int(atom_counts[idx[gid]]) != o_atoms[o_gid]:
            bad_atoms += 1
        if int(bond_counts[idx[gid]]) != o_bonds[o_gid]:
            bad_bonds += 1
    return {
        "n_graphs": len(kept),
        "n_components_not_one": bad_comp,
        "n_atom_count_mismatch": int(bad_atoms),
        "n_bond_count_mismatch": int(bad_bonds),
        "pass": bool(bad_comp == 0 and bad_atoms == 0 and bad_bonds == 0),
        "cache_content_fingerprint": cache["meta"].get("content_fingerprint"),
    }


# ---------------------------------------------------------------------------
# aggregation helpers (reuse the frozen Part-A statistics code)
# ---------------------------------------------------------------------------


def bootstrap_multiplicity(split: str, scaffold: Mapping[int, str]) -> dict[str, Any]:
    rng = np.random.default_rng(RC.BOOTSTRAP_SEED)
    rows = [int(x) for x in RC.SC.load_source("none")["rows"][split]]
    return RC.AN.cluster_multiplicity(rows, scaffold, rng=rng, replicates=RC.BOOTSTRAP_REPLICATES)


def molecule_mean(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[int, dict[str, float]]:
    return RC.AN.molecule_means(rows, tuple(keys))


def point_mean(values: Mapping[int, float]) -> float:
    return RC.AN.point_mean(values)


def ci95(reps: np.ndarray) -> list[float]:
    return RC.AN.pct_ci(reps)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(RC.TASKS), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--splits", default="valid,test")
    args = ap.parse_args(argv)
    task = args.task
    device = torch.device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir else RC.NEW_RUNS / task
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    t0 = time.perf_counter()
    RC.log(f"[{task}] evaluate: device={device} splits={splits}")

    spec = RC.frozen_spec()
    ck = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    target_mean = float(ck["target_norm"]["mean"])
    target_scale = float(ck["target_norm"]["scale"])
    model = RC.SC.DM.DissectLCRC(spec, RC.frozen_norm(), RC.SC.VARIANT)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.to(device)
    model.eval()
    if RC.SC.DM.parameter_count(model) != RC.SC.EXPECTED_PARAMS:
        raise RuntimeError("parameter count changed")

    labels = {str(r["graph_id"]): r for r in RC.read_csv(RC.NEW_ANALYSIS / "oracle_labels.csv")}
    cf = RC.read_csv(RC.NEW_ANALYSIS / "cf_isolation.csv")
    caliper = RC.read_csv(RC.SOURCE_ANALYSIS / "caliper_catalog.csv")
    n_c_by_pair: dict[str, int] = {}
    for r in caliper:
        n_c_by_pair[str(r["pair_id"])] = max(n_c_by_pair.get(str(r["pair_id"]), 0), 1)
    raw_rows = RC.SC.load_raw_rows()["rows"]["Lipophilicity"]
    prev = RC.SC.load_previous_inputs()
    scaffold = {rid: RC.G.bemis_murcko_key(str(raw_rows[rid]["smiles"])) for rid in range(RC.SC.N_MOLECULES)}

    checks: dict[str, Any] = {"task": task, "splits": {}}
    pair_rows: list[dict[str, Any]] = []
    predictions_by_split: dict[str, dict[str, float]] = {}

    for split in splits:
        cache, kept, _graph_rows = build_split_graph_cache(split, spec)
        vchk = _verify_cache(cache, kept)
        checks["splits"].setdefault(split, {})["graph_cache"] = vchk
        if not vchk["pass"]:
            raise SystemExit(f"CF_GRAPH_CACHE_INVALID: {split} {vchk}")
        pred_std = _predict_cache(model, cache, device)
        view_ids = [str(v) for v in cache["view_ids"]]
        pred_raw = pred_std * target_scale + target_mean
        pred_map = {gid: float(pred_raw[i]) for i, gid in enumerate(view_ids)}
        predictions_by_split[split] = pred_map

        for p in cf:
            if str(p["split"]) != split:
                continue
            pid = str(p["pair_id"])
            o_gid, t_gid, k_gid = str(p["o_graph_id"]), str(p["t_graph_id"]), str(p["k_graph_id"])
            if not all(g in pred_map and g in labels for g in (o_gid, t_gid, k_gid)):
                raise SystemExit(f"CF_GRAPH_MISSING: {pid}")
            pO, pT, pK = pred_map[o_gid], pred_map[t_gid], pred_map[k_gid]
            oO, oT, oK = (float(labels[o_gid][task]), float(labels[t_gid][task]), float(labels[k_gid][task]))
            dpT, dpK = pT - pO, pK - pO
            drT, drK = oT - oO, oK - oO
            aT, aK = abs(dpT), abs(dpK)
            rT, rK = abs(drT), abs(drK)
            row = {
                "task": task, "split": split, "family": str(p["family"]), "pair_id": pid,
                "row_id": int(p["row_id"]), "center": int(p["center"]), "target": int(p["target"]),
                "control": int(p["control"]), "n_c_candidates": int(p["n_c_candidates"]),
                "main_cf": int(p["main_cf"]),
                "pO": pO, "pT": pT, "pK": pK, "oO": oO, "oT": oT, "oK": oK,
                "delta_pred_T": dpT, "delta_pred_K": dpK, "delta_ref_T": drT, "delta_ref_K": drK,
                "abs_err_T": abs(dpT - drT), "abs_err_K": abs(dpK - drK),
                "abs_err_pair": 0.5 * (abs(dpT - drT) + abs(dpK - drK)),
                "baseline_T": rT, "baseline_K": rK, "baseline_pair": 0.5 * (rT + rK),
                "view_mae_T": abs(pT - oT), "view_mae_K": abs(pK - oK),
                "view_mse_T": (pT - oT) ** 2, "view_mse_K": (pK - oK) ** 2,
                "view_mae_pair": 0.5 * (abs(pT - oT) + abs(pK - oK)),
                "view_mse_pair": 0.5 * ((pT - oT) ** 2 + (pK - oK) ** 2),
                "sign_T": _sign_indicator(dpT, drT), "sign_K": _sign_indicator(dpK, drK),
                "zero_ref_T": int(rT <= RC.WC_LOG_CHANGE_TOL), "zero_ref_K": int(rK <= RC.WC_LOG_CHANGE_TOL),
                "q_pred": aT - aK, "q_ref": rT - rK, "q_err": abs((aT - aK) - (rT - rK)),
                "q_sign": _sign_indicator(aT - aK, rT - rK),
                "abs_dpred_T": aT, "abs_dpred_K": aK,
            }
            pair_rows.append(row)

    # ---- per-family aggregation with the shared scaffold bootstrap ------
    family_stats: dict[str, Any] = {}
    bonf_items: dict[str, Any] = {}
    q_lo = RC.ALPHA / (2.0 * RC.N_PRIMARY_ITEMS)
    q_hi = 1.0 - q_lo
    mult_by_split: dict[str, Any] = {}
    for split in splits:
        mult_by_split[split] = bootstrap_multiplicity(split, scaffold)
    for split in splits:
        mult = mult_by_split[split]
        for fam in RC.SC.FAMILIES:
            sel = [r for r in pair_rows if r["split"] == split and r["family"] == fam and r["main_cf"] == 1]
            sel_all = [r for r in pair_rows if r["split"] == split and r["family"] == fam]
            if not sel:
                family_stats[f"{task}/{split}/{fam}"] = {"n_pairs": 0, "n_molecules": 0, "NA": "no main_cf pair"}
                continue
            keys = ("abs_err_pair", "baseline_pair", "view_mae_pair", "view_mse_pair",
                    "sign_T", "sign_K", "q_err", "q_sign", "abs_dpred_T", "abs_dpred_K")
            mm = molecule_mean(sel, keys)
            gain = {rid: mm[rid].get("baseline_pair", np.nan) - mm[rid].get("abs_err_pair", np.nan) for rid in mm}
            reps_gain = RC.AN.bootstrap_mean(mult, gain)
            reps_mae = RC.AN.bootstrap_mean(mult, {rid: mm[rid].get("abs_err_pair", np.nan) for rid in mm})
            reps_base = RC.AN.bootstrap_mean(mult, {rid: mm[rid].get("baseline_pair", np.nan) for rid in mm})
            reps_sign = RC.AN.bootstrap_mean(mult, {rid: mm[rid].get("sign_T", np.nan) for rid in mm})
            # combined sign over T and K (mean of the two if both present)
            sign_comb = {}
            for rid, d in mm.items():
                vals = [d.get("sign_T", np.nan), d.get("sign_K", np.nan)]
                vals = [v for v in vals if np.isfinite(v)]
                sign_comb[rid] = float(np.mean(vals)) if vals else np.nan
            reps_sign_comb = RC.AN.bootstrap_mean(mult, sign_comb)
            reps_qerr = RC.AN.bootstrap_mean(mult, {rid: mm[rid].get("q_err", np.nan) for rid in mm})
            q_sign_comb = {rid: mm[rid].get("q_sign", np.nan) for rid in mm}
            reps_qsign = RC.AN.bootstrap_mean(mult, q_sign_comb)
            mse_mm = {rid: mm[rid].get("view_mse_pair", np.nan) for rid in mm}
            rmse_point = float(np.sqrt(point_mean(mse_mm))) if np.isfinite(point_mean(mse_mm)) else float("nan")
            reps_rmse = np.sqrt(RC.AN.bootstrap_mean(mult, mse_mm))
            n_views = 2 * len(sel)
            n_zero_ref = int(sum(r["zero_ref_T"] for r in sel) + sum(r["zero_ref_K"] for r in sel))
            abs_dp = np.asarray([r["abs_dpred_T"] for r in sel] + [r["abs_dpred_K"] for r in sel], dtype=np.float64)
            block = {
                "n_pairs": len(sel),
                "n_molecules": len(mm),
                "n_views": int(n_views),
                "n_pairs_all_main_queue": len(sel_all),
                "delta_mae": point_mean({rid: mm[rid].get("abs_err_pair", np.nan) for rid in mm}),
                "delta_mae_ci95": ci95(reps_mae),
                "baseline_zero_change": point_mean({rid: mm[rid].get("baseline_pair", np.nan) for rid in mm}),
                "baseline_ci95": ci95(reps_base),
                "gain": point_mean(gain),
                "gain_ci95": ci95(reps_gain),
                "gain_bonferroni": RC.AN.quantile_ci(reps_gain, q_lo, q_hi),
                "new_graph_mae_vs_own_oracle": point_mean({rid: mm[rid].get("view_mae_pair", np.nan) for rid in mm}),
                "new_graph_rmse_vs_own_oracle": rmse_point,
                "new_graph_rmse_ci95": ci95(reps_rmse),
                "sign_agreement": point_mean(sign_comb),
                "sign_agreement_ci95": ci95(reps_sign_comb),
                "sign_agreement_T_only": point_mean({rid: mm[rid].get("sign_T", np.nan) for rid in mm}),
                "n_views_zero_ref": int(n_zero_ref),
                "n_views_nonzero_ref": int(n_views - n_zero_ref),
                "q_abs_error": point_mean({rid: mm[rid].get("q_err", np.nan) for rid in mm}),
                "q_abs_error_ci95": ci95(reps_qerr),
                "q_sign_agreement": point_mean(q_sign_comb),
                "q_sign_agreement_ci95": ci95(reps_qsign),
                "n_pairs_nonzero_q_ref": int(sum(1 for r in sel if abs(r["q_ref"]) > RC.WC_LOG_CHANGE_TOL)),
                "abs_delta_pred": {
                    "mean": float(abs_dp.mean()), "median": float(np.median(abs_dp)),
                    "p95": float(np.percentile(abs_dp, 95)), "max": float(abs_dp.max()),
                    "mean_over_target_std": float(abs_dp.mean() / target_scale),
                    "median_over_target_std": float(np.median(abs_dp) / target_scale),
                    "p95_over_target_std": float(np.percentile(abs_dp, 95) / target_scale),
                    "max_over_target_std": float(abs_dp.max() / target_scale),
                },
            }
            family_stats[f"{task}/{split}/{fam}"] = block
            if task == "wc_logp" and split == "test":
                bonf_items[fam] = {
                    "gain": block["gain"], "ci95": block["gain_ci95"],
                    "bonferroni": block["gain_bonferroni"],
                    "n_molecules": block["n_molecules"], "n_pairs": block["n_pairs"],
                    "gain_bonferroni_excludes_zero": bool(
                        np.isfinite(block["gain_bonferroni"][0]) and np.isfinite(block["gain_bonferroni"][1])
                        and (block["gain_bonferroni"][0] > 0 or block["gain_bonferroni"][1] < 0)
                    ),
                }

    # ---- C-matched (calibration) fixed secondary scope ------------------
    calib_stats: dict[str, Any] = {}
    for split in splits:
        mult = mult_by_split[split]
        for fam in RC.SC.FAMILIES:
            sel = [r for r in pair_rows if r["split"] == split and r["family"] == fam
                   and r["main_cf"] == 1 and r["n_c_candidates"] > 0]
            if not sel:
                calib_stats[f"{split}/{fam}"] = {"n_pairs": 0}
                continue
            mm = molecule_mean(sel, ("abs_err_pair", "baseline_pair", "view_mae_pair"))
            gain = {rid: mm[rid].get("baseline_pair", np.nan) - mm[rid].get("abs_err_pair", np.nan) for rid in mm}
            calib_stats[f"{split}/{fam}"] = {
                "n_pairs": len(sel), "n_molecules": len(mm),
                "delta_mae": point_mean({rid: mm[rid].get("abs_err_pair", np.nan) for rid in mm}),
                "baseline": point_mean({rid: mm[rid].get("baseline_pair", np.nan) for rid in mm}),
                "gain": point_mean(gain),
                "gain_ci95": ci95(RC.AN.bootstrap_mean(mult, gain)),
                "new_graph_mae_vs_own_oracle": point_mean({rid: mm[rid].get("view_mae_pair", np.nan) for rid in mm}),
            }

    # ---- original-graph metrics (recomputed from the frozen original caches) ----
    train_cache = RC.load_original_cache("train")
    valid_cache = RC.load_original_cache("valid")
    test_cache = RC.load_original_cache("test")
    train_lab = TT_labels(task, "train", train_cache)
    valid_lab = TT_labels(task, "valid", valid_cache)
    test_lab = TT_labels(task, "test", test_cache)
    orig_train = raw_metrics(_predict_cache(model, train_cache, device), train_lab, target_mean, target_scale)
    orig_valid = raw_metrics(_predict_cache(model, valid_cache, device), valid_lab, target_mean, target_scale)
    orig_test = raw_metrics(_predict_cache(model, test_cache, device), test_lab, target_mean, target_scale)
    test_train_mean_rmse = float(np.sqrt(np.mean((test_lab - target_mean) ** 2)))
    # cross-check against the numbers written by the training stage
    train_json_path = out_dir / "train.json"
    train_json = RC.read_json(train_json_path) if train_json_path.exists() else None
    origin_graph_crosscheck = None
    if train_json is not None:
        b = train_json["metrics"]["best"]
        origin_graph_crosscheck = {
            "train_rmse_diff": abs(b["train"]["rmse"] - orig_train["rmse"]),
            "valid_rmse_diff": abs(b["valid"]["rmse"] - orig_valid["rmse"]),
            "test_rmse_diff": abs(b["test"]["rmse"] - orig_test["rmse"]),
            "pass": bool(max(abs(b["train"]["rmse"] - orig_train["rmse"]),
                              abs(b["valid"]["rmse"] - orig_valid["rmse"]),
                              abs(b["test"]["rmse"] - orig_test["rmse"])) <= RC.REPRO_TOL),
        }

    metrics = {
        "round_id": RC.ROUND_ID,
        "protocol_id": RC.PROTOCOL_ID,
        "task": task,
        "task_description": RC.TASK_DESCRIPTION[task],
        "units": RC.TASK_UNITS[task],
        "cf_scope": PRIMARY_SCOPE,
        "target_norm": {"mean": target_mean, "scale": target_scale, "source": "Train original labels only"},
        "original_graph_metrics": {
            "train": orig_train, "valid": orig_valid, "test": orig_test,
            "train_mean_constant_baseline_test_rmse": test_train_mean_rmse,
            "train_target_std": target_scale,
            "note": "raw units; Train mean constant baseline predicts the Train target mean",
            "recomputed_in_evaluate": True,
            "crosscheck_vs_train_json": origin_graph_crosscheck,
        },
        "bootstrap": {"seed": RC.BOOTSTRAP_SEED, "replicates": RC.BOOTSTRAP_REPLICATES,
                      "cluster": "Bemis-Murcko scaffold, one shared multiplicity per split"},
        "bonferroni": {"n_items": RC.N_PRIMARY_ITEMS, "alpha": RC.ALPHA, "quantiles": [q_lo, q_hi],
                       "items": bonf_items},
        "family_stats": family_stats,
        "c_matched_secondary": calib_stats,
        "counts": {
            "n_pairs_total": len(pair_rows),
            "n_pairs_main_cf": int(sum(r["main_cf"] for r in pair_rows)),
            "n_pairs_supplement": int(sum(1 - r["main_cf"] for r in pair_rows)),
        },
        "checks": checks,
    }
    RC.write_json(RC.NEW_ANALYSIS / f"partB_{task}_metrics.json", metrics)
    cols = ["task", "split", "family", "pair_id", "row_id", "center", "target", "control", "n_c_candidates",
            "main_cf", "pO", "pT", "pK", "oO", "oT", "oK", "delta_pred_T", "delta_pred_K", "delta_ref_T",
            "delta_ref_K", "abs_err_T", "abs_err_K", "abs_err_pair", "baseline_T", "baseline_K", "baseline_pair",
            "view_mae_T", "view_mae_K", "view_mae_pair", "sign_T", "sign_K", "zero_ref_T", "zero_ref_K",
            "q_pred", "q_ref", "q_err", "q_sign", "abs_dpred_T", "abs_dpred_K"]
    RC.write_csv(RC.NEW_ANALYSIS / f"partB_{task}_pairs.csv", pair_rows, cols)
    runtime = {"task": task, "device": str(device), "splits": splits,
               "wall_seconds": float(time.perf_counter() - t0), "environment": RC.env_info(),
               "git_commit": RC.git_commit(), "git_dirty": RC.git_dirty()}
    RC.write_json(RC.NEW_ANALYSIS / f"partB_{task}_runtime.json", runtime)
    RC.log(f"[{task}] evaluate done in {runtime['wall_seconds']:.1f}s; "
           f"pairs={len(pair_rows)}")
    return 0


def _sign_indicator(delta_pred: float, delta_ref: float) -> float:
    if not np.isfinite(delta_pred) or not np.isfinite(delta_ref):
        return float("nan")
    if abs(delta_ref) <= RC.WC_LOG_CHANGE_TOL:
        return float("nan")
    return float(np.sign(delta_pred) == np.sign(delta_ref))


if __name__ == "__main__":
    raise SystemExit(main())
