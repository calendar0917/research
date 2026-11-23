"""Post-training analysis for RICE-CC-v0.

Everything is evaluated on **frozen** best-Dev checkpoints.  No retraining and no
Dev-based model selection.  The primary contrast is

    Delta = RMSE(CC_NULL, D0 Dev) - RMSE(CC_REAL, D0 Dev)

with a paired molecule-level bootstrap over the same 504 Dev rows.  The
frozen-checkpoint switch from ``J_real`` to ``J_base`` is an off-manifold
intervention, not a retrained control.

Outputs under ``analysis/``:
  arms.csv, curves.csv, summary.json, coupling.json,
  frozen_intervention.json, verdict.json
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import rice_common as C
import rice_data as D
import rice_model as M
import rice_structure as S
import rice_tensor as T

SIZES = (2, 3, 4)

# Atomic-number-index -> symbol for the common OGB elements.  Unknown indices
# are reported as ``Z<index>``.  Field 0 is the atomic-number *index*, not Z.
ELEMENTS = {0: "*", 1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
            8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si",
            15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 26: "Fe",
            29: "Cu", 30: "Zn", 33: "As", 34: "Se", 35: "Br", 53: "I"}


def element(index: int) -> str:
    return ELEMENTS.get(int(index), f"Z{int(index)}")


def _mix(values: np.ndarray, dims: Sequence[int]) -> np.ndarray:
    out = np.zeros(values.shape[0], np.int64)
    for i, dim in enumerate(dims):
        out = out * int(dim) + values[:, i].astype(np.int64)
    return out


# ---------------------------------------------------------------------------
# basic statistics
# ---------------------------------------------------------------------------

def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(pred, np.float64) - y) ** 2)))


def _mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, np.float64) - y)))


def bootstrap_delta(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray,
                    n_boot: int, seed: int) -> dict[str, Any]:
    """Paired molecule bootstrap of RMSE(a) - RMSE(b); positive => b better."""
    rng = np.random.default_rng(int(seed))
    n = int(y.size)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    sq_a = (pred_a[idx] - y[idx]) ** 2
    sq_b = (pred_b[idx] - y[idx]) ** 2
    delta = np.sqrt(sq_a.mean(axis=1)) - np.sqrt(sq_b.mean(axis=1))
    point = float(_rmse(pred_a, y) - _rmse(pred_b, y))
    return {"point": point,
            "ci_low": float(np.percentile(delta, 2.5)),
            "ci_high": float(np.percentile(delta, 97.5)),
            "fraction_positive": float((delta > 0).mean()),
            "n_boot": int(n_boot), "seed": int(seed),
            "unit": "molecule",
            "sign": "RMSE(CC_NULL) - RMSE(CC_REAL); positive = CC_REAL better",
            "ci_scope": "Dev molecule sampling uncertainty only; not training-seed variance"}


def _summary_values(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, np.float64).reshape(-1)
    if x.size == 0:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "abs_mean": float(np.abs(x).mean()), "abs_max": float(np.abs(x).max()),
            "positive_fraction": float((x > 0).mean()),
            "negative_fraction": float((x < 0).mean())}


def _weighted_between(groups: Sequence[np.ndarray], values: np.ndarray) -> dict[str, float]:
    """Nested variance decomposition for a scalar per bond."""
    x = np.asarray(values, np.float64)
    n = x.size
    total = float(x.var())
    between = 0.0
    within = 0.0
    for idx in groups:
        idx = np.asarray(idx, np.int64)
        if idx.size == 0:
            continue
        m = float(x[idx].mean())
        between += float(idx.size) * (m - float(x.mean())) ** 2
        within += float(((x[idx] - m) ** 2).sum())
    between /= max(1, n)
    within /= max(1, n)
    return {"total_variance": total, "between_variance": between,
            "within_variance": within,
            "between_fraction": between / total if total > 0 else 0.0}


# ---------------------------------------------------------------------------
# per-bond coupling diagnostics
# ---------------------------------------------------------------------------

def collect_coupling(model: M.RiceModel, structs: Mapping[int, S.MoleculeStruct],
                     rows: Sequence[int], bundle: D.DataBundle, device: Any = "cpu",
                     batch: int = 256) -> dict[str, Any]:
    """Collect delta and the bond / endpoint metadata on one split."""
    rows = np.asarray(rows, np.int64)
    delta_parts: list[np.ndarray] = []
    base_abs: list[np.ndarray] = []
    real_abs: list[np.ndarray] = []
    changed: list[np.ndarray] = []
    bond_type: list[np.ndarray] = []
    bond_full: list[np.ndarray] = []
    elem_pair: list[np.ndarray] = []
    atom_pair_full: list[np.ndarray] = []
    atom_dims = bundle.atom_dims
    bond_dims = bundle.bond_dims
    model.eval()
    with torch.no_grad():
        for start in range(0, rows.size, int(batch)):
            part = rows[start:start + int(batch)]
            sub = {int(r): structs[int(r)] for r in part}
            coll = T.collate(part.tolist(), sub, bundle.raw, device)
            out = model(coll, arm="CC_REAL", return_aux=True)
            d = out["delta"].detach().cpu().numpy()
            jb = out["j_base"].detach().cpu().numpy()
            jr = out["j"].detach().cpu().numpy()
            bf = coll["bond_feat"].cpu().numpy()
            af = coll["atom_feat"].cpu().numpy()
            bi = coll["bond_index"].cpu().numpy()
            delta_parts.append(d.astype(np.float64))
            base_abs.append(np.abs(jb).max(axis=-1).astype(np.float64))
            real_abs.append(np.abs(jr).max(axis=-1).astype(np.float64))
            changed.append((np.abs(jr - jb).max(axis=-1) > 1e-8).astype(np.int8))
            bond_type.append(bf[:, 0].astype(np.int64))
            bond_full.append(_mix(bf, bond_dims))
            # endpoint element pair (sorted atomic-number indices)
            e0 = af[bi[:, 0], 0].astype(np.int64)
            e1 = af[bi[:, 1], 0].astype(np.int64)
            elem_pair.append(np.stack([np.minimum(e0, e1), np.maximum(e0, e1)], axis=1))
            # endpoint full atom-category pair (sorted mixed-radix codes)
            ac = _mix(af, atom_dims)
            a0 = ac[bi[:, 0]]
            a1 = ac[bi[:, 1]]
            atom_pair_full.append(np.stack([np.minimum(a0, a1), np.maximum(a0, a1)], axis=1))
    delta = np.concatenate(delta_parts, axis=0) if delta_parts else np.zeros((0, 16))
    signed = delta.mean(axis=1) if delta.size else np.zeros(0)
    absmean = np.abs(delta).mean(axis=1) if delta.size else np.zeros(0)
    norm = np.linalg.norm(delta, axis=1) if delta.size else np.zeros(0)
    return {
        "rows": rows,
        "delta": delta,
        "signed_mean": signed,
        "abs_mean": absmean,
        "norm": norm,
        "base_absmax": np.concatenate(base_abs) if base_abs else np.zeros(0),
        "real_absmax": np.concatenate(real_abs) if real_abs else np.zeros(0),
        "changed": np.concatenate(changed) if changed else np.zeros(0, np.int8),
        "bond_type": np.concatenate(bond_type) if bond_type else np.zeros(0, np.int64),
        "bond_full": np.concatenate(bond_full) if bond_full else np.zeros(0, np.int64),
        "elem_pair": np.concatenate(elem_pair, axis=0) if elem_pair else np.zeros((0, 2), np.int64),
        "atom_pair_full": np.concatenate(atom_pair_full, axis=0) if atom_pair_full
        else np.zeros((0, 2), np.int64),
    }


def _group_report(data: Mapping[str, Any], group_key: np.ndarray,
                  name: str, min_pair_count: int = 5) -> dict[str, Any]:
    x = np.asarray(data["signed_mean"], np.float64)
    a = np.asarray(data["abs_mean"], np.float64)
    key = np.asarray(group_key, np.int64)
    out: dict[str, Any] = {}
    for value in sorted(np.unique(key).tolist()):
        idx = np.nonzero(key == value)[0]
        if idx.size == 0:
            continue
        out[str(int(value))] = {
            "n_bonds": int(idx.size),
            "signed_mean": _summary_values(x[idx]),
            "abs_mean": _summary_values(a[idx]),
            "frac_abs_mean_gt_threshold": float(np.mean(a[idx] > 0.001)),
        }
    return out


def _endpoint_variation(data: Mapping[str, Any], bond_type: np.ndarray,
                        endpoint_pair: np.ndarray, value: np.ndarray,
                        min_pair_count: int = 5) -> dict[str, Any]:
    """How much coupling variation remains *within* one bond type due to endpoints."""
    x = np.asarray(value, np.float64)
    btype = np.asarray(bond_type, np.int64)
    ep = np.asarray(endpoint_pair, np.int64)
    per_type: dict[str, Any] = {}
    for bt in sorted(np.unique(btype).tolist()):
        sel = np.nonzero(btype == bt)[0]
        if sel.size == 0:
            continue
        x_sel = x[sel]
        pairs = ep[sel]
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        groups = [np.nonzero(inverse == g)[0] for g in range(unique_pairs.shape[0])]
        groups_stable = [g for g in groups if g.size >= int(min_pair_count)]
        pair_means = [float(x_sel[g].mean()) for g in groups]
        dec_all = _weighted_between(groups, x_sel)
        dec_stable = _weighted_between(groups_stable, x_sel) if groups_stable else {
            "total_variance": float(x_sel.var()), "between_variance": 0.0,
            "within_variance": float(x_sel.var()), "between_fraction": 0.0}
        coverage = (float(sum(g.size for g in groups_stable)) / sel.size
                    if groups_stable else 0.0)
        per_type[str(int(bt))] = {
            "n_bonds": int(sel.size),
            "n_endpoint_pairs": int(unique_pairs.shape[0]),
            "n_stable_pairs_min_count": int(len(groups_stable)),
            "stable_pair_coverage": coverage,
            "pair_mean_range": float(max(pair_means) - min(pair_means)) if pair_means else 0.0,
            "all_pairs": dec_all,
            "stable_pairs": dec_stable,
        }
    # nested overall decomposition
    bt_groups = [np.nonzero(btype == bt)[0] for bt in sorted(np.unique(btype).tolist())]
    between_bt = _weighted_between(bt_groups, x)
    # within-bond-type endpoint decomposition, weighted by bond-type size
    within_bt_between = 0.0
    within_bt_within = 0.0
    for bt, g in zip(sorted(np.unique(btype).tolist()), bt_groups):
        sel = g
        x_sel = x[sel]
        pairs = ep[sel]
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        groups = [np.nonzero(inverse == k)[0] for k in range(unique_pairs.shape[0])]
        dec = _weighted_between(groups, x_sel)
        within_bt_between += dec["between_variance"] * sel.size
        within_bt_within += dec["within_variance"] * sel.size
    n = max(1, x.size)
    overall_dec = {
        "total_variance": float(x.var()),
        "between_bond_type": between_bt["between_variance"],
        "between_endpoint_within_bond_type": within_bt_between / n,
        "within_endpoint_pair": within_bt_within / n,
    }
    total = overall_dec["total_variance"]
    overall_dec["between_bond_type_fraction"] = (
        overall_dec["between_bond_type"] / total if total > 0 else 0.0)
    overall_dec["between_endpoint_within_bond_type_fraction"] = (
        overall_dec["between_endpoint_within_bond_type"] / total if total > 0 else 0.0)
    overall_dec["within_endpoint_pair_fraction"] = (
        overall_dec["within_endpoint_pair"] / total if total > 0 else 0.0)
    return {"overall": overall_dec, "per_bond_type": per_type,
            "min_pair_count": int(min_pair_count),
            "value": "signed delta averaged over the 16 channels",
            "endpoint_pair": "sorted endpoint atomic-number-index pair"}


def coupling_report(data: Mapping[str, Any], split_name: str) -> dict[str, Any]:
    delta = np.asarray(data["delta"], np.float64)
    flat = delta.reshape(-1) if delta.size else np.zeros(0)
    signed = np.asarray(data["signed_mean"], np.float64)
    absmean = np.asarray(data["abs_mean"], np.float64)
    report = {
        "split": split_name,
        "n_molecules": int(data["rows"].size),
        "n_bonds": int(delta.shape[0]),
        "n_channels": int(delta.shape[1]) if delta.ndim == 2 else 0,
        "overall": {
            "delta_all_channels": _summary_values(flat),
            "delta_signed_mean_per_bond": _summary_values(signed),
            "delta_abs_mean_per_bond": _summary_values(absmean),
            "delta_norm_per_bond": _summary_values(data["norm"]),
            "fraction_bonds_with_abs_mean_gt_0.001":
                float(np.mean(absmean > 0.001)) if absmean.size else 0.0,
            "fraction_bonds_with_changed_J_real_vs_J_base":
                float(np.mean(np.asarray(data["changed"]) > 0)) if signed.size else 0.0,
            "j_base_abs_max": float(np.max(data["base_absmax"])) if signed.size else 0.0,
            "j_real_abs_max": float(np.max(data["real_absmax"])) if signed.size else 0.0,
        },
        "per_bond_type": _group_report(data, data["bond_type"], "bond_type"),
        "per_full_bond_category": _group_report(data, data["bond_full"],
                                                "full_bond_category"),
        "endpoint_combination_variation_element_pair": _endpoint_variation(
            data, data["bond_type"], data["elem_pair"], signed),
        "endpoint_combination_variation_full_atom_pair": _endpoint_variation(
            data, data["bond_type"], data["atom_pair_full"], signed),
    }
    # Readable endpoint pair labels for the top element pairs.
    ep = np.asarray(data["elem_pair"], np.int64)
    if ep.size:
        pairs, counts = np.unique(ep, axis=0, return_counts=True)
        order = np.argsort(-counts)
        report["endpoint_element_pairs_top"] = [
            {"elements": f"{element(pairs[i, 0])}-{element(pairs[i, 1])}",
             "atomic_number_indices": [int(pairs[i, 0]), int(pairs[i, 1])],
             "n_bonds": int(counts[i])}
            for i in order[:20]
        ]
    return report


# ---------------------------------------------------------------------------
# loading and evaluation
# ---------------------------------------------------------------------------

def _load_run(arm: str, results_dir: Path) -> dict[str, Any]:
    run_dir = results_dir / f"{arm}_S0"
    manifest = C.read_json(run_dir / "manifest.json")
    dev = np.load(run_dir / "predictions_best_dev.npz")
    fit = np.load(run_dir / "predictions_best_fit.npz")
    state = torch.load(run_dir / "state_best.pt", map_location="cpu", weights_only=False)
    return {"arm": arm, "dir": run_dir, "manifest": manifest, "dev": dev, "fit": fit,
            "state": state}


def predict_split(model: M.RiceModel, structs: Mapping[int, S.MoleculeStruct],
                  rows: Sequence[int], bundle: D.DataBundle, arm: str,
                  device: Any = "cpu", batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(rows, np.int64)
    model.eval()
    out_std: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, rows.size, int(batch)):
            part = rows[start:start + int(batch)]
            sub = {int(r): structs[int(r)] for r in part}
            coll = T.collate(part.tolist(), sub, bundle.raw, device)
            out_std.append(model(coll, arm=arm)["yhat_std"].detach().cpu().numpy())
    pred_std = np.concatenate(out_std) if out_std else np.zeros(0)
    pred_raw = pred_std * float(bundle.scaler["std"]) + float(bundle.scaler["mean"])
    return pred_raw, pred_std


def _read_curves(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = C.load_config()
    bundle = D.load_bundle()
    dev_rows = np.asarray(bundle.eval_rows, np.int64)
    y_dev = bundle.labels[dev_rows]
    hist = cfg["historical_d0_rmse"]
    checks_path = C.ANALYSIS_DIR / "checks.json"
    checks = C.read_json(checks_path) if checks_path.exists() else {"all_pass": False,
                                                                    "n_checks": 0}
    runs = {arm: _load_run(arm, C.RESULTS_DIR) for arm in ("CC_NULL", "CC_REAL")}
    null_run, real_run = runs["CC_NULL"], runs["CC_REAL"]
    if not np.array_equal(null_run["dev"]["row_id"], real_run["dev"]["row_id"]):
        raise RuntimeError("Dev row order differs between the two arms")
    if not np.array_equal(null_run["dev"]["row_id"], dev_rows):
        raise RuntimeError("Dev predictions do not follow the frozen D0 Dev row order")

    arms = []
    for arm, run in runs.items():
        m = run["manifest"]
        arms.append({
            "arm": arm, "pair_potential": m.get("coupling", "?"),
            "params": m["params"]["total"],
            "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"],
            "fit_rmse": m["train_rmse_at_best"], "fit_mae": m["train_mae_at_best"],
            "dev_rmse": m["dev_rmse"], "dev_mae": m["dev_mae"],
            "train_seconds": m["train_seconds"], "wall_seconds": m["wall_seconds"],
            "gpu_hours": m["gpu_hours"], "prep_seconds": m["prep_seconds"],
            "peak_vram_bytes": m["peak_vram_bytes"],
            "n_fit": m["n_fit"], "n_dev": m["n_dev"],
            "init_hashes": m["init_hashes"], "data_hashes": m["data_hashes"],
        })
    C.write_csv(C.ANALYSIS_DIR / "arms.csv", arms)

    # combined full curves
    curve_rows = []
    for arm in ("CC_NULL", "CC_REAL"):
        for row in _read_curves(runs[arm]["dir"] / "curves.csv"):
            row = dict(row)
            row["arm"] = arm
            curve_rows.append(row)
    C.write_csv(C.ANALYSIS_DIR / "curves.csv", curve_rows)

    n_boot = int(cfg["analysis"]["bootstrap_resamples"])
    boot_seed = int(cfg["analysis"]["bootstrap_seed"])
    delta = bootstrap_delta(null_run["dev"]["pred"], real_run["dev"]["pred"],
                            y_dev, n_boot, boot_seed)
    null_dev = float(null_run["manifest"]["dev_rmse"])
    real_dev = float(real_run["manifest"]["dev_rmse"])
    null_mae = float(null_run["manifest"]["dev_mae"])
    real_mae = float(real_run["manifest"]["dev_mae"])

    hist_null = float(hist["RICE_CUMULANT_S0"])
    null_repro = {
        "reference_dev_rmse": hist_null,
        "cc_null_dev_rmse": null_dev,
        "difference": null_dev - hist_null,
        "abs_difference": abs(null_dev - hist_null),
        "tolerance": float(cfg["verdict"]["null_reproduction_tolerance_rmse"]),
        "ok": abs(null_dev - hist_null) <= float(cfg["verdict"]
                                                 ["null_reproduction_tolerance_rmse"]),
        "note": ("CC_NULL is mathematically identical to historical RICE_CUMULANT; the "
                 "tolerance covers GPU/early-stopping numerical fluctuation only. A larger "
                 "deviation would make the pair incomparable until investigated."),
    }
    data_identity = {
        "null_fit_rows_sha256": null_run["manifest"]["fit_rows_sha256"],
        "real_fit_rows_sha256": real_run["manifest"]["fit_rows_sha256"],
        "null_dev_rows_sha256": null_run["manifest"]["dev_rows_sha256"],
        "real_dev_rows_sha256": real_run["manifest"]["dev_rows_sha256"],
        "null_split_sha256": null_run["manifest"]["data_hashes"]["split"]["split_sha256"],
        "real_split_sha256": real_run["manifest"]["data_hashes"]["split"]["split_sha256"],
        "same_fit": null_run["manifest"]["fit_rows_sha256"]
        == real_run["manifest"]["fit_rows_sha256"],
        "same_dev": null_run["manifest"]["dev_rows_sha256"]
        == real_run["manifest"]["dev_rows_sha256"],
        "same_split": null_run["manifest"]["data_hashes"]["split"]["split_sha256"]
        == real_run["manifest"]["data_hashes"]["split"]["split_sha256"],
        "n_fit": int(null_run["manifest"]["n_fit"]),
        "n_dev": int(null_run["manifest"]["n_dev"]),
        "scaler_null": null_run["manifest"]["scaler"],
        "scaler_real": real_run["manifest"]["scaler"],
        "same_scaler": null_run["manifest"]["scaler"] == real_run["manifest"]["scaler"],
        "init_hashes_equal": null_run["manifest"]["init_hashes"]
        == real_run["manifest"]["init_hashes"],
        "params_equal": null_run["manifest"]["params"] == real_run["manifest"]["params"],
    }

    # ---- coupling diagnostics (CC_REAL checkpoint, Dev and Fit) -------------
    model = M.load_model_from_state(real_run["state"]["state"], bundle.atom_dims,
                                    bundle.bond_dims, D.normalisers(bundle), config=cfg,
                                    device=args.device)
    model.eval()
    structs = S.build_all(bundle, np.union1d(bundle.fit_rows, bundle.eval_rows))
    dev_coupling_raw = collect_coupling(model, structs, dev_rows, bundle, device=args.device)
    fit_coupling_raw = collect_coupling(model, structs, bundle.fit_rows, bundle,
                                        device=args.device)
    coupling = {"dev": coupling_report(dev_coupling_raw, "dev"),
                "fit": coupling_report(fit_coupling_raw, "fit"),
                "note": ("delta is the signed per-channel correction on each physical "
                         "bond; endpoint-combination variance is a descriptive "
                         "decomposition, not a causal effect")}
    C.write_json(C.ANALYSIS_DIR / "coupling.json", coupling)

    # non-degeneracy uses Dev only
    dev_flat = np.asarray(dev_coupling_raw["delta"], np.float64).reshape(-1)
    dev_absmean = np.asarray(dev_coupling_raw["abs_mean"], np.float64)
    nondegenerate = {
        "max_abs_delta": float(np.abs(dev_flat).max()) if dev_flat.size else 0.0,
        "std_delta": float(dev_flat.std()) if dev_flat.size else 0.0,
        "fraction_abs_gt_threshold": float(np.mean(np.abs(dev_flat)
                                                   > float(cfg["verdict"]
                                                           ["coupling_min_max_abs_delta"])))
        if dev_flat.size else 0.0,
        "fraction_bonds_abs_mean_gt_threshold": float(np.mean(dev_absmean
                                                             > float(cfg["verdict"]
                                                                     ["coupling_min_max_abs_delta"])))
        if dev_absmean.size else 0.0,
        "fraction_bonds_changed": float(np.mean(
            np.asarray(dev_coupling_raw["changed"]) > 0)) if dev_absmean.size else 0.0,
        "ok": bool(dev_flat.size
                   and float(np.abs(dev_flat).max())
                   > float(cfg["verdict"]["coupling_min_max_abs_delta"])
                   and float(dev_flat.std())
                   > float(cfg["verdict"]["coupling_min_std_delta"])
                   and float(np.mean(np.abs(dev_flat)
                                     > float(cfg["verdict"]["coupling_min_max_abs_delta"])))
                   > float(cfg["verdict"]["coupling_min_abs_delta_fraction"])),
    }

    # ---- frozen off-manifold intervention --------------------------------
    pred_chem, _ = predict_split(model, structs, dev_rows, bundle, "CC_REAL",
                                 device=args.device)
    pred_switch, _ = predict_split(model, structs, dev_rows, bundle, "CC_NULL",
                                   device=args.device)
    intervention = {
        "label": "off_manifold_frozen_intervention",
        "checkpoint": str((real_run["dir"] / "state_best.pt").relative_to(C.REPO_ROOT)),
        "dev_rmse_cc_real": _rmse(pred_chem, y_dev),
        "dev_rmse_switched_to_cc_null": _rmse(pred_switch, y_dev),
        "delta_rmse_switch_minus_real": _rmse(pred_switch, y_dev) - _rmse(pred_chem, y_dev),
        "dev_mae_cc_real": _mae(pred_chem, y_dev),
        "dev_mae_switched": _mae(pred_switch, y_dev),
        "mean_abs_prediction_change": float(np.abs(pred_switch - pred_chem).mean()),
        "max_abs_prediction_change": float(np.abs(pred_switch - pred_chem).max()),
        "n_rows": int(dev_rows.size),
        "note": ("Same CC_REAL weights, pair formula replaced by J_base at evaluation "
                 "time. This is off the trained input manifold and cannot replace the "
                 "retrained CC_NULL vs CC_REAL contrast."),
    }
    C.write_json(C.ANALYSIS_DIR / "frozen_intervention.json", intervention)

    # ---- summary -----------------------------------------------------------
    summary = {
        "protocol_id": cfg["protocol_id"], "version": cfg["version"],
        "config_sha256": C.sha256_json(cfg),
        "checks": {"all_pass": bool(checks.get("all_pass")),
                   "n_checks": int(checks.get("n_checks", 0)),
                   "n_fail": int(checks.get("n_fail", 0)),
                   "path": str(checks_path.relative_to(C.REPO_ROOT))},
        "arms": arms,
        "main": {
            "delta_definition": "RMSE(CC_NULL, D0 Dev) - RMSE(CC_REAL, D0 Dev)",
            "delta": delta,
            "cc_null_dev_rmse": null_dev, "cc_null_dev_mae": null_mae,
            "cc_real_dev_rmse": real_dev, "cc_real_dev_mae": real_mae,
            "cc_null_fit_rmse": float(null_run["manifest"]["train_rmse_at_best"]),
            "cc_real_fit_rmse": float(real_run["manifest"]["train_rmse_at_best"]),
            "cc_null_best_epoch": int(null_run["manifest"]["best_epoch"]),
            "cc_real_best_epoch": int(real_run["manifest"]["best_epoch"]),
        },
        "historical": {
            "rice_cumulant_s0": hist_null,
            "r2_sentence_s0": float(hist["R2_SENTENCE_S0"]),
            "r2_sentence_s1": float(hist["R2_SENTENCE_S1"]),
            "gine": float(hist["GINE_REF"]),
            "cc_real_minus_rice_cumulant": real_dev - hist_null,
            "cc_real_minus_r2_s0": real_dev - float(hist["R2_SENTENCE_S0"]),
            "cc_real_minus_r2_s1": real_dev - float(hist["R2_SENTENCE_S1"]),
            "cc_real_minus_gine": real_dev - float(hist["GINE_REF"]),
            "cc_null_minus_rice_cumulant": null_dev - hist_null,
        },
        "null_reproduction": null_repro,
        "data_identity": data_identity,
        "nondegenerate_coupling": nondegenerate,
        "frozen_intervention": intervention,
        "gpu_hours_both_arms": float(null_run["manifest"]["gpu_hours"]
                                     + real_run["manifest"]["gpu_hours"]),
        "prep_seconds_total": float(null_run["manifest"]["prep_seconds"]
                                    + real_run["manifest"]["prep_seconds"]),
        "seed": 0, "split": "D0", "arms_count": 2,
    }
    C.write_json(C.ANALYSIS_DIR / "summary.json", summary)

    # ---- verdict -----------------------------------------------------------
    invalid_reasons: list[str] = []
    if not checks.get("all_pass"):
        invalid_reasons.append("pre-training checks did not all pass")
    if not null_repro["ok"]:
        invalid_reasons.append("CC_NULL Dev RMSE deviates from historical RICE_CUMULANT "
                               "beyond the frozen tolerance")
    for key in ("same_fit", "same_dev", "same_split", "same_scaler",
                "init_hashes_equal", "params_equal"):
        if not data_identity[key]:
            invalid_reasons.append(f"arm consistency failed: {key}")
    if not nondegenerate["ok"]:
        invalid_reasons.append("chemistry-conditioned coupling is degenerate on Dev")
    if invalid_reasons:
        label = cfg["verdict"]["label_invalid"]
    else:
        promising = (delta["point"]
                     >= float(cfg["verdict"]["promising_min_delta_raw_minus_chem"])
                     and (delta["ci_low"] > 0.0
                          if cfg["verdict"]["promising_requires_ci_low_positive"]
                          else True)
                     and real_dev <= float(cfg["verdict"]["promising_max_chem_dev_rmse"])
                     and (nondegenerate["ok"]
                          if cfg["verdict"]["promising_requires_nondegenerate_coupling"]
                          else True))
        if promising:
            label = cfg["verdict"]["label_promising"]
        elif delta["point"] > 0.0:
            label = cfg["verdict"]["label_weak"]
        else:
            label = cfg["verdict"]["label_no_gain"]
    verdict = {
        "label": label,
        "delta": delta,
        "cc_null_dev_rmse": null_dev, "cc_real_dev_rmse": real_dev,
        "delta_positive": bool(delta["point"] > 0.0),
        "cc_real_at_or_below_0.575": bool(real_dev <= 0.575),
        "ci_low_positive": bool(delta["ci_low"] > 0.0),
        "nondegenerate_coupling": nondegenerate,
        "null_reproduction": null_repro,
        "data_identity": data_identity,
        "checks_all_pass": bool(checks.get("all_pass")),
        "invalid_reasons": invalid_reasons,
        "criteria": dict(cfg["verdict"]),
        "historical": summary["historical"],
        "stop_condition": ("two arms, seed 0, D0 only, no D1, no official Valid/Test, "
                           "no hyper-parameter search"),
        "interpretation": ("CHEM_BINDING_PROMISING only motivates a separate D1 "
                           "confirmation; it is not a final success claim."),
    }
    C.write_json(C.ANALYSIS_DIR / "verdict.json", verdict)

    print(f"verdict: {label}")
    print(f"CC_NULL dev {null_dev:.4f}  CC_REAL dev {real_dev:.4f}  "
          f"Delta {delta['point']:+.4f} [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]")
    print(f"nondegenerate={nondegenerate['ok']}  null_repro={null_repro['ok']}  "
          f"checks={checks.get('all_pass')}")
    if invalid_reasons:
        print("INVALID reasons:", invalid_reasons)


if __name__ == "__main__":
    main()