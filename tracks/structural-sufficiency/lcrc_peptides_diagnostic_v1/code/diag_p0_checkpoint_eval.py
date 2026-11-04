"""P0.3 — re-evaluate the three frozen Fast v1 checkpoints on Train and Validation.

Only Train and Validation are read.  ``sys.addaudithook`` raises if any Test
artifact is opened, so a Test read cannot happen silently.  The historical
checkpoints are loaded strictly (``strict=True``); nothing is retrained.

Usage (GPU box, from the repository root):

    python tracks/structural-sufficiency/lcrc_peptides_diagnostic_v1/code/diag_p0_checkpoint_eval.py \
        --seeds 0 1 2 --device cuda --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import diag_common as C

C.install_test_access_guard()

import bp_data as D  # noqa: E402
import bp_metrics as MET  # noqa: E402
import bp_model as BM  # noqa: E402
import bp_train as T  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def per_label_ap(y: np.ndarray, probs: np.ndarray) -> list[float]:
    out = []
    for j in range(y.shape[1]):
        out.append(float(average_precision_score(y[:, j], probs[:, j])))
    return out


def evaluate_seed(seed: int, device: torch.device) -> dict[str, Any]:
    ckpt_path = C.history_checkpoint_path(seed)
    if not ckpt_path.exists():
        return {"seed": seed, "status": "CHECKPOINT_UNAVAILABLE", "checkpoint_path": C.track_rel(ckpt_path)}
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    record = C.history_run(seed)
    ckpt_sha = C.sha256_file(ckpt_path)
    if ckpt_sha != record["checkpoint_sha256"]:
        raise RuntimeError(f"seed {seed}: checkpoint sha256 differs from the historical record")

    data = T._make_task("peptide", C.DATASET, seed)
    spec_hash = data.spec.hash()
    if ckpt["feature_spec_hash"] != spec_hash:
        raise RuntimeError(f"seed {seed}: feature spec hash {ckpt['feature_spec_hash']} != {spec_hash}")
    if ckpt["norm_fingerprint"] != data.norm["norm_fingerprint"]:
        raise RuntimeError(f"seed {seed}: norm fingerprint mismatch")
    cache_train_sha = C.sha256_file(C.PREV_CACHE / "peptides_func_train.pt")
    cache_valid_sha = C.sha256_file(C.PREV_CACHE / "peptides_func_valid.pt")
    if ckpt["cache_identity"]["cache_train_sha256"] != cache_train_sha:
        raise RuntimeError(f"seed {seed}: train cache sha256 mismatch")
    if ckpt["cache_identity"]["cache_valid_sha256"] != cache_valid_sha:
        raise RuntimeError(f"seed {seed}: valid cache sha256 mismatch")

    model, info = BM.build_c1(
        data.spec,
        data.norm,
        int(data.n_out),
        int(seed),
        device,
        cont_slice=data.cont_slice,
        cont_mean=np.asarray(data.cont_norm.get("mean", []), dtype=np.float32),
        cont_scale=np.asarray(data.cont_norm.get("scale", []), dtype=np.float32),
    )
    state = {k: v.to(device) for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    structural = {
        "parameters": int(info["parameters"]),
        "historical_parameters": int(ckpt["model_info"]["parameters"]),
        "parameters_match": int(info["parameters"]) == int(ckpt["model_info"]["parameters"]),
        "local_layout_match": info["local_layout"] == ckpt["model_info"]["local_layout"],
        "rho_input_dim_match": int(info["rho_input_dim"]) == int(ckpt["model_info"]["rho_input_dim"]),
        "local_input_dim_match": int(info["local_input_dim"]) == int(ckpt["model_info"]["local_input_dim"]),
        "shared_with_full_parameters": ckpt["model_info"]["shared_with_full"]["all_equal"],
        "init_state_hash_reproduced": info["init_state_hash"] == ckpt["model_info"]["init_state_hash"],
        "n_state_tensors_loaded": len(ckpt["state_dict"]),
    }
    if not (
        structural["parameters_match"]
        and structural["local_layout_match"]
        and structural["rho_input_dim_match"]
        and structural["init_state_hash_reproduced"]
    ):
        raise RuntimeError(f"seed {seed}: rebuilt model differs structurally from the frozen checkpoint: {structural}")

    t0 = time.perf_counter()
    train_logits = T.predict_logits(model, data.train_cache, data.train_rows, device)
    valid_logits = T.predict_logits(model, data.valid_cache, data.valid_rows, device)
    seconds = time.perf_counter() - t0

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return (1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))).astype(np.float64)

    train_probs = _sigmoid(train_logits)
    valid_probs = _sigmoid(valid_logits)
    y_train = np.asarray(data.y_train, dtype=np.float64)
    y_valid = np.asarray(data.y_valid, dtype=np.float64)
    if train_probs.shape != y_train.shape or valid_probs.shape != y_valid.shape:
        raise RuntimeError(f"seed {seed}: prediction/label shape mismatch")
    if not (
        np.isfinite(train_probs).all()
        and np.isfinite(valid_probs).all()
        and (train_probs >= 0).all()
        and (train_probs <= 1).all()
        and (valid_probs >= 0).all()
        and (valid_probs <= 1).all()
    ):
        raise RuntimeError(f"seed {seed}: probabilities outside [0,1]")

    train_ap = per_label_ap(y_train, train_probs)
    valid_ap = per_label_ap(y_valid, valid_probs)
    valid_macro = float(np.mean(valid_ap))
    train_macro = float(np.mean(train_ap))
    stored_macro = float(ckpt["best_metric"])
    # independent re-derivation through the shared metric helper
    helper = MET.macro_ap(y_valid, valid_probs.astype(np.float32))
    helper_macro = float(helper["value"])
    delta_helper = abs(valid_macro - helper_macro)
    if delta_helper > 1e-9:
        raise RuntimeError(f"seed {seed}: macro AP recomputation mismatch {delta_helper}")

    Path(seed_out_dir(seed)).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(seed_out_dir(seed)) / "train_valid_predictions.npz",
        train_probs=train_probs,
        valid_probs=valid_probs,
        y_train=y_train,
        y_valid=y_valid,
        labels=np.asarray(C.LABELS),
    )
    rows = []
    train_pos_rate = y_train.mean(axis=0)
    valid_pos_rate = y_valid.mean(axis=0)
    for j, name in enumerate(C.LABELS):
        rows.append(
            {
                "seed": seed,
                "label_index": j,
                "label": name,
                "train_pos_rate": float(train_pos_rate[j]),
                "valid_pos_rate": float(valid_pos_rate[j]),
                "train_ap": train_ap[j],
                "valid_ap": valid_ap[j],
                "valid_ap_pct": 100.0 * valid_ap[j],
                "valid_ap_minus_valid_pos_rate": valid_ap[j] - float(valid_pos_rate[j]),
                "valid_constant_baseline_ap": float(valid_pos_rate[j]),
            }
        )
    return {
        "seed": seed,
        "status": "CHECKPOINT_EVALUATED",
        "checkpoint_path": C.track_rel(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "best_epoch": int(ckpt["best_epoch"]),
        "epochs_run": int(ckpt["epochs_run"]),
        "stored_best_valid_macro_ap": stored_macro,
        "stored_best_valid_macro_ap_pct": 100.0 * stored_macro,
        "train_macro_ap": train_macro,
        "valid_macro_ap": valid_macro,
        "valid_macro_ap_pct": 100.0 * valid_macro,
        "valid_macro_ap_helper_recheck": helper_macro,
        "valid_macro_ap_helper_delta": delta_helper,
        "valid_minus_stored_best": valid_macro - stored_macro,
        "train_minus_valid_macro_ap": train_macro - valid_macro,
        "structural": structural,
        "eval_seconds": seconds,
        "per_label": rows,
        "test_access_attempts": 0,
    }


def seed_out_dir(seed: int) -> Path:
    return Path(OUT_DIR) / f"seed{seed}"


OUT_DIR = "."


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(C.LEGACY_SEEDS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    global OUT_DIR
    OUT_DIR = args.out_dir
    device = torch.device(args.device)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        C.log(f"evaluating frozen Fast v1 checkpoint seed {seed}")
        res = evaluate_seed(seed, device)
        seed_dir = seed_out_dir(seed)
        seed_dir.mkdir(parents=True, exist_ok=True)
        rows = res.pop("per_label", [])
        C.write_json(seed_dir / "eval.json", res)
        label_rows.extend(rows)
        summaries.append({k: v for k, v in res.items() if k != "structural"})
        C.write_json(seed_dir / "structural.json", res["structural"])
        C.log(
            f"seed {seed}: valid macro AP {100*res['valid_macro_ap']:.2f}% "
            f"(stored {100*res['stored_best_valid_macro_ap']:.2f}%), train macro AP {100*res['train_macro_ap']:.2f}%"
        )

    write_csv(C.RESULTS / "p0_frozen_checkpoints_summary.csv", summaries)
    write_csv(C.RESULTS / "p0_frozen_checkpoints_per_label.csv", label_rows)
    C.write_json(
        C.ANALYSIS / "p0_frozen_checkpoints_audit.json",
        {
            "round_id": C.ROUND_ID,
            "generated_utc": C.utcnow(),
            "per_seed_run_dir": C.track_rel(Path(OUT_DIR)),
            "device": str(device),
            "seeds": args.seeds,
            "test_access_guard": "sys.addaudithook blocks peptides_func_test.pt / peptides_func_y_test.npy / test_predictions.npz",
            "test_artifacts_opened": 0,
            "environment": C.environment(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
