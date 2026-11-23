"""Replay the two formal checkpoints and re-derive their reported metrics.

Independent of the training process: loads `state_best.pt` from each run
directory, rebuilds the split structures, predicts Fit and held-out rows and
compares against the manifest metrics and the committed prediction files.
Also records the descriptive D0 seed spread (seed 1 vs the GWIC-v0 seed 0
checkpoint on the same 504 molecules).

Writes ``analysis/replay.json``.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np
import torch

import r2_common as RC
import r2_data as RD
import gwic_data as GD
import gwic_model as M
import gwic_structure as S
import gwic_tensor as T

TOL = 1e-5


def _predict(model: M.GwicModel, rows: np.ndarray, structs: dict,
             raw: dict, scaler: dict, batch: int = 256) -> np.ndarray:
    model.eval()
    out = np.zeros(rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, rows.size, batch):
            part = rows[start:start + batch]
            tensors = T.collate(part.tolist(), structs, raw, "cpu")
            out[start:start + part.size] = (
                model(tensors, arm="SENTENCE_BASE")["yhat_std"].double().numpy())
    return out * float(scaler["std"]) + float(scaler["mean"])


def _replay(run_id: str, split: str, seed: int) -> dict[str, Any]:
    run_dir = RC.RESULTS_DIR / run_id
    manifest = RC.read_json(run_dir / "manifest.json")
    bundle = RD.load_bundle(split, manifest["atom_scalar_mode"])
    atom_dims, bond_dims = RD.GD.atom_bond_dims()
    ckpt = torch.load(run_dir / "state_best.pt", map_location="cpu", weights_only=False)
    model = M.load_model_from_state(atom_dims, bond_dims, ckpt["state"],
                                    RD.norm_of(bundle), use_words=False,
                                    config=RC.frozen_config(), device="cpu")
    rows = np.union1d(bundle.fit_rows, bundle.eval_rows)
    structs = S.build_all(bundle.raw, rows)
    pred_fit = _predict(model, bundle.fit_rows, structs, bundle.raw, manifest["scaler"])
    pred_dev = _predict(model, bundle.eval_rows, structs, bundle.raw, manifest["scaler"])
    y_fit = bundle.labels[bundle.fit_rows]
    y_dev = bundle.labels[bundle.eval_rows]
    stored_dev = np.load(run_dir / "predictions_best_dev.npz")
    stored_fit = np.load(run_dir / "predictions_best_fit.npz")
    out = {
        "run_id": run_id, "split": split, "seed": int(seed),
        "atom_scalar_mode": manifest["atom_scalar_mode"],
        "n_fit": int(bundle.fit_rows.size), "n_dev": int(bundle.eval_rows.size),
        "dev_rmse_replay": RC.rmse(pred_dev, y_dev),
        "dev_rmse_manifest": float(manifest["dev_rmse"]),
        "dev_rmse_abs_delta": abs(RC.rmse(pred_dev, y_dev) - float(manifest["dev_rmse"])),
        "dev_mae_replay": RC.mae(pred_dev, y_dev),
        "dev_mae_manifest": float(manifest["dev_mae"]),
        "fit_rmse_replay": RC.rmse(pred_fit, y_fit),
        "fit_rmse_manifest": float(manifest["train_rmse_at_best"]),
        "max_abs_delta_vs_stored_dev": float(np.max(np.abs(pred_dev - stored_dev["pred"]))),
        "max_abs_delta_vs_stored_fit": float(np.max(np.abs(pred_fit - stored_fit["pred"]))),
        "stored_row_ids_match_dev": bool(np.array_equal(stored_dev["row_id"], bundle.eval_rows)),
        "best_epoch": int(manifest["best_epoch"]),
        "epochs_run": int(manifest["epochs_run"]),
        "dev_rmse_at_final_epoch": float(manifest["row"]["dev_rmse"]
                                         if "row" in manifest else float("nan")),
    }
    out["ok"] = bool(out["dev_rmse_abs_delta"] < 1e-6
                     and abs(out["dev_mae_replay"] - out["dev_mae_manifest"]) < 1e-6
                     and abs(out["fit_rmse_replay"] - out["fit_rmse_manifest"]) < 1e-6
                     and out["max_abs_delta_vs_stored_dev"] < TOL
                     and out["max_abs_delta_vs_stored_fit"] < TOL)
    del structs
    preds = {"fit": pred_fit, "dev": pred_dev, "rows_fit": bundle.fit_rows,
             "rows_dev": bundle.eval_rows}
    return {"summary": out, "preds": preds}


def _seed_spread(d0_replay: dict[str, Any]) -> dict[str, Any]:
    frozen = np.load(RC.GWIC_RESULTS / "SENTENCE_BASE_S0" / "predictions_best_dev.npz")
    rows = d0_replay["preds"]["rows_dev"]
    order = {int(r): i for i, r in enumerate(frozen["row_id"].tolist())}
    idx = np.asarray([order[int(r)] for r in rows.tolist()], np.int64)
    other = frozen["pred"][idx]
    mine = d0_replay["preds"]["dev"]
    y = frozen["label"][idx]
    return {
        "n": int(rows.size),
        "dev_rmse_seed0_gwic_v0": RC.rmse(other, y),
        "dev_rmse_seed1": RC.rmse(mine, y),
        "dev_rmse_delta_seed1_minus_seed0": float(RC.rmse(mine, y) - RC.rmse(other, y)),
        "prediction_correlation": float(np.corrcoef(mine, other)[0, 1]),
        "mean_abs_prediction_difference": float(np.mean(np.abs(mine - other))),
        "max_abs_prediction_difference": float(np.max(np.abs(mine - other))),
        "mean_abs_error_seed0": float(np.mean(np.abs(other - y))),
        "mean_abs_error_seed1": float(np.mean(np.abs(mine - y))),
        "note": ("seed 0 prediction comes from the GWIC-v0 SENTENCE_BASE_S0 checkpoint "
                 "(v0 atom-scalar normalisation); the difference mixes seed and normaliser"),
    }


def _split_statistics() -> dict[str, Any]:
    """Per-split Fit statistics, recomputed here so the manifests stay auditable."""
    raw = GD._load_raw()
    atom_dims, bond_dims = GD.atom_bond_dims()
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    out: dict[str, Any] = {}
    supports: dict[str, Any] = {}
    for name in ("D0", "D1"):
        spec = RD.load_split(name, raw)
        fit = spec["fit_rows"]
        atom_ok, bond_ok = GD.category_support(raw, fit, atom_dims, bond_dims)
        supports[name] = (atom_ok, bond_ok)
        out[name] = {
            "n_fit": int(fit.size),
            "n_fit_atoms": int(np.diff(node_ptr)[fit].sum()),
            "n_fit_bonds": int(np.diff(bond_ptr)[fit].sum()),
            "fit_rows_sha256": spec["spec"]["fit_rows_sha256"],
            "atom_support_sizes": [int(x.sum()) for x in atom_ok],
            "bond_support_sizes": [int(x.sum()) for x in bond_ok],
        }
    a0, b0 = supports["D0"]
    a1, b1 = supports["D1"]
    out["category_support_sets_identical_across_splits"] = bool(
        all(np.array_equal(x, y) for x, y in zip(a0, a1))
        and all(np.array_equal(x, y) for x, y in zip(b0, b1)))
    out["note"] = ("support masks are recomputed from each run's Fit rows; with 2856 Fit "
                   "molecules the realised OGB vocabulary happens to coincide, while the "
                   "Fit atom/bond counts and all target/graph/atom normalisers differ")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RC.ANALYSIS_DIR / "replay.json"))
    args = ap.parse_args(argv)
    cfg = RC.load_config()
    records = []
    d0 = None
    for run in cfg["runs"]:
        out = _replay(run["run_id"], run["split"], int(run["seed"]))
        records.append(out["summary"])
        if run["split"] == "D0":
            d0 = out
        print(f"{run['run_id']}: replay dev_rmse={out['summary']['dev_rmse_replay']:.6f} "
              f"(manifest {out['summary']['dev_rmse_manifest']:.6f}) ok={out['summary']['ok']}")
    payload = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": RC.now_utc(),
        "git_commit": RC.git_commit(),
        "runs": records,
        "all_ok": all(r["ok"] for r in records),
        "split_statistics": _split_statistics(),
        "d0_seed_spread": _seed_spread(d0) if d0 is not None else {},
    }
    RC.write_json(args.out, payload)
    print(f"all_ok={payload['all_ok']} -> {args.out}")
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())