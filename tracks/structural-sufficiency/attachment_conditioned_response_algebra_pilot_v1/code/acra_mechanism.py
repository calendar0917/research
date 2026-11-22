"""Mechanism diagnostics for the fitted ACRA checkpoints (11.2).

Reads the best checkpoint of each arm, never re-trains.  Reports term scales,
cancellation, response sign structure, whether the interaction / relation terms
are actually used (prediction change when zeroed), per-branch gradient norms,
and the dependence of local response size on Train support frequency.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import acra_common as C
import acra_data as D
import acra_cache as K
import acra_model as M

RUNS = {
    "ACRA_ADD": "ACRA_ADD_S0",
    "ACRA_COND": "ACRA_COND_S0",
    "ACRA_REL": "ACRA_REL_S0",
}


def load_model(run_id: str, pack: Mapping[str, Any], dims, device: str = "cpu"):
    import torch

    path = C.run_dir(run_id) / "state_best.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = M.ACRAModel(cfg, dims[0], dims[1], ckpt["arm"])
    model.load_pack(pack)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt


def event_outputs(model, source: D.BatchSource, rows: Sequence[int],
                  batch_size: int = 128) -> dict[str, np.ndarray]:
    import torch

    rows = np.asarray(rows, np.int64)
    keys = ("r", "scalar_s", "scalar_c", "interaction", "ev_mol")
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = source.make(part)
            res = model.forward_batch(batch)
            for k in keys:
                if k == "ev_mol":
                    arr = res["ev_mol"].detach().cpu().numpy() + start
                else:
                    arr = res[k].detach().double().cpu().numpy()
                out[k].append(arr)
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def _term_stats(out: Mapping[str, np.ndarray], y_std: np.ndarray) -> dict[str, Any]:
    terms = {k: np.asarray(out[k], np.float64) for k in ("e_atom", "e_bond", "e_rel")}
    yhat = np.asarray(out["yhat_std"], np.float64)
    residual = yhat - y_std
    total = sum(np.abs(v) for v in terms.values())
    payload: dict[str, Any] = {
        "n": int(y_std.size),
        "mean_yhat_std": float(np.mean(yhat)),
        "std_yhat_std": float(np.std(yhat)),
        "mean_abs_residual": float(np.mean(np.abs(residual))),
        "rmse_std": float(np.sqrt(np.mean(residual ** 2))),
        "beta_mean": float(np.mean(out["beta"])),
        "cancellation_index": float(1.0 - np.mean(np.abs(yhat - out["beta"]))
                                    / max(1e-12, np.mean(total))),
        "mean_abs_beta_plus_terms": float(np.mean(np.abs(yhat - out["beta"]))),
    }
    for name, value in terms.items():
        payload[name] = {
            "mean": float(np.mean(value)),
            "std": float(np.std(value)),
            "mean_abs": float(np.mean(np.abs(value))),
            "p95_abs": float(np.percentile(np.abs(value), 95)),
            "share_of_abs_terms": float(np.mean(np.abs(value)) / max(1e-12, np.mean(total))),
            "pearson_with_y_std": C.pearson(value, y_std),
            "sign_positive_fraction": float(np.mean(value > 0)),
        }
    return payload


def _ablation(model, source: D.BatchSource, rows: np.ndarray, y_std: np.ndarray,
              what: str) -> dict[str, Any]:
    import torch

    base = M.predict_rows(model, source, rows, "cpu")
    saved = (model.use_interaction, model.use_relation)
    if what == "interaction":
        if not model.use_interaction:
            return {"defined": False}
        model.use_interaction = False
    elif what == "relation":
        if not model.use_relation:
            return {"defined": False}
        model.use_relation = False
    else:
        raise ValueError(what)
    try:
        without = M.predict_rows(model, source, rows, "cpu")
    finally:
        model.use_interaction, model.use_relation = saved
    delta = without["yhat_std"] - base["yhat_std"]
    base_rmse = float(np.sqrt(np.mean((base["yhat_std"] - y_std) ** 2)))
    without_rmse = float(np.sqrt(np.mean((without["yhat_std"] - y_std) ** 2)))
    return {
        "defined": True,
        "base_rmse_std": base_rmse,
        "without_rmse_std": without_rmse,
        "delta_rmse_std": without_rmse - base_rmse,
        "mean_abs_delta_yhat": float(np.mean(np.abs(delta))),
        "max_abs_delta_yhat": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "pearson_delta_with_base": C.pearson(delta, base["yhat_std"]),
    }


def _gradient_norms(model, source: D.BatchSource, rows: np.ndarray,
                    y_std: np.ndarray, seed: int = 0) -> dict[str, float]:
    import torch

    torch.manual_seed(seed)
    model.train()
    batch = source.make(rows)
    y = torch.as_tensor(y_std, dtype=torch.float32)
    out = model.forward_batch(batch)
    terms = model.loss(out, batch, y)
    model.zero_grad(set_to_none=True)
    terms["loss"].backward()
    groups: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0.0) + float(param.grad.detach().pow(2).sum())
    model.eval()
    return {k: float(np.sqrt(v)) for k, v in groups.items()}


def run(cfg: Mapping[str, Any], log=print) -> dict[str, Any]:
    pack = K.ensure_pack(cfg, log=log)
    raw = D.load_raw(cfg)
    dims = C.official_feature_dims()
    splits = D.official_splits(raw)
    labels = np.asarray(raw["labels"], np.float64)
    scaler = C.target_scaler(labels, splits["train"])
    y_std = (labels - scaler["mean"]) / scaler["std"]
    source = D.BatchSource(pack, "cpu")

    ev_ptr = np.asarray(pack["ev_ptr"], np.int64)
    ev_mol = np.repeat(np.arange(int(pack["n_molecules"])), np.diff(ev_ptr))
    mol_split = np.full(int(pack["n_molecules"]), 2, np.uint8)
    mol_split[splits["train"]] = 0
    mol_split[splits["valid"]] = 1
    split = mol_split[ev_mol]
    combo_ids = np.asarray(pack["ev_combo"], np.int64)
    train_combo = np.bincount(combo_ids[split == 0], minlength=int(pack["combo_s"].size))
    support_of_event = train_combo[combo_ids]
    buckets = [(0, 0, "0_unseen"), (1, 1, "1"), (2, 4, "2-4"), (5, 9, "5-9"),
               (10, 49, "10-49"), (50, 10 ** 9, "50+")]

    out: dict[str, Any] = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack.get("cache_sha256"),
        "definitions": {
            "cancellation_index": "1 - mean|yhat-beta| / mean(|E_atom|+|E_bond|+|E_rel|); "
                                  "1 means complete cancellation, 0 no cancellation",
            "interaction_ablation": "same checkpoint with the bilinear interaction term set to 0",
            "relation_ablation": "same checkpoint with E_rel set to 0",
            "support_bucket": "number of Train directed events with the same combined local "
                              "identity (S, C, cut-bond attributes)",
        },
        "arms": {},
    }
    for arm, run_id in RUNS.items():
        path = C.run_dir(run_id) / "state_best.pt"
        if not path.is_file():
            log(f"  {arm}: checkpoint missing at {path}; skipped")
            continue
        model, ckpt = load_model(run_id, pack, dims)
        rec: dict[str, Any] = {
            "run_id": run_id,
            "best_epoch": int(ckpt["epoch"]),
            "arm": arm,
            "params": ckpt.get("ledger", model.parameter_ledger()),
            "test_labels_read": False,
        }
        for split_name in ("train", "valid"):
            rows = np.asarray(splits[split_name], np.int64)
            pred = M.predict_rows(model, source, rows, "cpu")
            rec[split_name] = _term_stats(pred, y_std[rows])
            ev = event_outputs(model, source, rows)
            r = ev["r"]
            scalar = ev["scalar_s"] + ev["scalar_c"]
            rec[split_name]["responses"] = {
                "n_directed_events": int(r.size),
                "r_quantiles": C.quantiles(r, (0, 5, 25, 50, 75, 95, 100)),
                "positive_fraction": float(np.mean(r > 0)),
                "negative_fraction": float(np.mean(r < 0)),
                "positive_mass": float(np.sum(np.clip(r, 0, None))),
                "negative_mass": float(np.sum(np.clip(r, None, 0))),
                "mass_cancellation": float(1.0 - abs(np.sum(r))
                                           / max(1e-12, np.sum(np.abs(r)))),
                "scalar_quantiles": C.quantiles(scalar, (5, 50, 95)),
            }
            if model.use_interaction:
                inter = ev["interaction"]
                rec[split_name]["interaction"] = {
                    "mean_abs": float(np.mean(np.abs(inter))),
                    "p95_abs": float(np.percentile(np.abs(inter), 95)),
                    "ratio_to_scalar_abs": float(np.mean(np.abs(inter))
                                                 / max(1e-12, np.mean(np.abs(scalar)))),
                    "pearson_with_r_remainder": C.pearson(inter, scalar),
                    "ablation": _ablation(model, source, rows, y_std[rows], "interaction"),
                }
            if model.use_relation:
                rec[split_name]["relation"] = {
                    "gate": float(np.asarray(
                        __import__("torch").sigmoid(model.gate_raw).detach().cpu())),
                    "ablation": _ablation(model, source, rows, y_std[rows], "relation"),
                }
            split_code = {"train": 0, "valid": 1}[split_name]
            bucket_rec = {}
            for lo, hi, label in buckets:
                mask = (support_of_event[split == split_code] >= lo) & \
                       (support_of_event[split == split_code] <= hi)
                if not mask.any():
                    bucket_rec[label] = {"n": 0}
                    continue
                vals = r[mask]
                abs_scalar = np.abs(scalar[mask])
                entry = {
                    "n": int(mask.sum()),
                    "fraction_of_events": float(np.mean(mask)),
                    "mean_abs_r": float(np.mean(np.abs(vals))),
                    "std_r": float(np.std(vals)),
                    "mean_abs_scalar_sum": float(np.mean(abs_scalar)),
                    "positive_fraction": float(np.mean(vals > 0)),
                }
                if model.use_interaction:
                    entry["mean_abs_interaction"] = float(np.mean(np.abs(ev["interaction"][mask])))
                bucket_rec[label] = entry
            rec[split_name]["support_buckets"] = bucket_rec
        if "train" in rec and rec["train"].get("support_buckets"):
            rec["rare_vs_frequent_response_abs"] = {
                "singleton_mean_abs_r": rec["train"]["support_buckets"].get("1", {}).get(
                    "mean_abs_r", 0.0),
                "frequent_mean_abs_r": rec["train"]["support_buckets"].get("50+", {}).get(
                    "mean_abs_r", 0.0),
                "note": "Train events always have support >= 1, so the singleton bucket is "
                        "the rarest non-empty stratum; '0_unseen' is empty by definition",
            }
        rows = np.asarray(splits["train"], np.int64)[:64]
        rec["gradient_norms_train_batch"] = _gradient_norms(model, source, rows, y_std[rows])
        # per-molecule error vs local novelty (Valid)
        valid_rows = np.asarray(splits["valid"], np.int64)
        pred_v = M.predict_rows(model, source, valid_rows, "cpu")
        novelty = []
        errors = []
        for i, row in enumerate(valid_rows.tolist()):
            ids = combo_ids[ev_ptr[row]:ev_ptr[row + 1]]
            novelty.append(float(np.mean(train_combo[ids] == 0)) if ids.size else 0.0)
            errors.append(abs(float(pred_v["yhat_std"][i] - y_std[row])))
        rec["valid_novelty_vs_error"] = {
            "pearson_novelty_abs_error": C.pearson(np.asarray(novelty), np.asarray(errors)),
            "mean_abs_error_low_novelty": float(np.mean([e for n, e in zip(novelty, errors)
                                                         if n <= 0.1])),
            "mean_abs_error_high_novelty": float(np.mean([e for n, e in zip(novelty, errors)
                                                          if n > 0.5])),
            "n_high_novelty": int(sum(1 for n in novelty if n > 0.5)),
        }
        out["arms"][arm] = rec
        log(f"  {arm}: valid rmse_std={rec['valid']['rmse_std']:.4f} "
            f"cancel={rec['valid']['cancellation_index']:.3f}")
    C.write_json(C.analysis_dir() / "contribution_diagnostics.json", out)
    log("wrote contribution_diagnostics.json")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    _args = ap.parse_args(argv)
    run(C.load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())