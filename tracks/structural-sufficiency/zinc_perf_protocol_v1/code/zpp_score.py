"""Terminal one-shot official-test scoring after protocol freeze.

This is the only script in the namespace that constructs or labels the official
test split. It refuses to run without a matching freeze, verifies every frozen
checkpoint, and writes a read lock so the test is not silently re-read.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import zpp_common as C
import zpp_data as D
import zpp_freeze as F
import zpp_p3 as P3
import zpp_zsrt as Z

C.ensure_p3_modules()
import structural_relation_objects_v1 as sro  # noqa: E402
import esr_models as M  # noqa: E402

LOCK = C.PREDICTIONS / "test_read.lock"


def _claim_lock() -> None:
    C.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    try:
        with open(LOCK, "x", encoding="utf-8") as fh:
            fh.write(f"{C.utcnow()} {C.git_commit()}\n")
    except FileExistsError:
        raise RuntimeError(f"test read lock already exists: {LOCK}")


def _predict_p3(seed: int, checkpoint: Path, device):
    prep = C.read_json(C.P3_PREP)
    model = M.build_arm("B", int(seed), prep, None, 0.0, device)
    M.load_encoder(model, int(seed), C.P3_SSL_S0 if seed == 0 else C.P3_SSL_S1, device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict({k: v.to(device) for k, v in ckpt["state_dict"].items()})
    valid_pred, valid_mae = P3.eval_objects(model, D.load_p3_valid_objects(),
                                            D.split_labels(D.build_data("valid", rrwp=False)), device)
    test_ds = D.pyg_zinc("test")
    test_objs = [
        sro.build_mol_objects(i, test_ds[i].x.numpy(), test_ds[i].edge_index.numpy(),
                              test_ds[i].edge_attr.numpy())
        for i in range(1000)
    ]
    y_test = np.asarray([float(test_ds[i].y) for i in range(1000)], dtype=np.float64)
    test_pred, test_mae = P3.eval_objects(model, test_objs, y_test, device)
    return valid_pred, valid_mae, test_pred, test_mae, int(ckpt["epoch"])


def _predict_zsrt(variant: dict, seed: int, checkpoint: Path, device):
    model, _, n_params = Z.build_model(variant, int(seed), device)
    model.to(torch.device(device))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    Z._load_state(model, ckpt["state_dict"], device)
    valid_data = D.build_data("valid", ksteps=variant["ksteps"], rrwp=variant["rrwp"])
    test_data = D.build_data("test", ksteps=variant["ksteps"], rrwp=variant["rrwp"])
    y_valid = D.split_labels(valid_data)
    y_test = D.split_labels(test_data)
    valid_pred, valid_mae = Z.evaluate(model, valid_data, y_valid, device, split="valid")
    test_pred, test_mae = Z.evaluate(model, test_data, y_test, device, split="test")
    return valid_pred, valid_mae, test_pred, test_mae, int(ckpt["epoch"]), n_params


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "per_seed": [float(v) for v in values],
        "mean": float(arr.mean()),
        "sample_std": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "n_seeds": int(arr.size),
    }


def score(device: str, force: bool = False) -> dict:
    frozen = F.verify_freeze()
    if LOCK.exists() and not force:
        raise RuntimeError(f"test already read in this namespace: {LOCK}")
    if not LOCK.exists():
        _claim_lock()

    variant = frozen["variant"]
    y_test_p3 = np.asarray([float(D.pyg_zinc("test")[i].y) for i in range(1000)], dtype=np.float64)
    test_data = D.build_data("test", ksteps=variant["ksteps"], rrwp=variant["rrwp"])
    y_test_z = D.split_labels(test_data)
    if not np.array_equal(y_test_p3, y_test_z):
        raise RuntimeError("P3 and ZSRT test label order differ")

    pred_dir = C.PREDICTIONS
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = {"p3": {}, "zsrt": {}}

    for seed_str, row in frozen["checkpoints"]["p3"].items():
        seed = int(seed_str)
        ckpt = C.REPO / row["path"]
        vpred, vmae, tpred, tmae, epoch = _predict_p3(seed, ckpt, device)
        if abs(vmae - row["best_valid_mae"]) > 1e-5:
            raise RuntimeError(f"P3 seed {seed} best-valid replay mismatch: {vmae} vs {row['best_valid_mae']}")
        np.savez(pred_dir / f"valid_p3_s{seed}.npz", row=np.arange(len(vpred)), pred=vpred)
        np.savez(pred_dir / f"test_p3_s{seed}.npz", row=np.arange(len(tpred)), pred=tpred)
        rows["p3"][seed_str] = {
            "best_epoch": epoch,
            "valid_mae": float(vmae),
            "test_mae": float(tmae),
            "test_rmse": C.rmse(tpred, y_test_p3),
            "checkpoint_sha256": row["sha256"],
            "test_pred_sha256": C.sha256_array(tpred),
            "parameter_count": row["parameter_count"],
            "updates": row["updates"],
            "wall_seconds": row["wall_seconds"],
        }
        C.log(f"[score p3 s{seed}] best_ep={epoch} valid={vmae:.6f} test={tmae:.6f}")

    for seed_str, row in frozen["checkpoints"]["zsrt"].items():
        seed = int(seed_str)
        ckpt = C.REPO / row["path"]
        vpred, vmae, tpred, tmae, epoch, n_params = _predict_zsrt(variant, seed, ckpt, device)
        if abs(vmae - row["best_valid_mae"]) > 1e-5:
            raise RuntimeError(f"ZSRT seed {seed} best-valid replay mismatch: {vmae} vs {row['best_valid_mae']}")
        np.savez(pred_dir / f"valid_zsrt_s{seed}.npz", row=np.arange(len(vpred)), pred=vpred)
        np.savez(pred_dir / f"test_zsrt_s{seed}.npz", row=np.arange(len(tpred)), pred=tpred)
        rows["zsrt"][seed_str] = {
            "best_epoch": epoch,
            "valid_mae": float(vmae),
            "test_mae": float(tmae),
            "test_rmse": C.rmse(tpred, y_test_z),
            "checkpoint_sha256": row["sha256"],
            "test_pred_sha256": C.sha256_array(tpred),
            "parameter_count": int(n_params),
            "updates": row["updates"],
            "wall_seconds": row["wall_seconds"],
        }
        C.log(f"[score zsrt s{seed}] best_ep={epoch} valid={vmae:.6f} test={tmae:.6f}")

    p3_valid = [rows["p3"][s]["valid_mae"] for s in sorted(rows["p3"], key=int)]
    p3_test = [rows["p3"][s]["test_mae"] for s in sorted(rows["p3"], key=int)]
    z_valid = [rows["zsrt"][s]["valid_mae"] for s in sorted(rows["zsrt"], key=int)]
    z_test = [rows["zsrt"][s]["test_mae"] for s in sorted(rows["zsrt"], key=int)]
    shared = sorted(set(rows["p3"]) & set(rows["zsrt"]), key=int)
    paired_valid = {s: rows["zsrt"][s]["valid_mae"] - rows["p3"][s]["valid_mae"] for s in shared}
    paired_test = {s: rows["zsrt"][s]["test_mae"] - rows["p3"][s]["test_mae"] for s in shared}

    out = {
        "protocol": C.PROTOCOL_ID,
        "status": "TERMINAL_TEST_READ_ONCE",
        "config_name": frozen["config_name"],
        "variant": variant,
        "zsrt_seeds": frozen["zsrt_seeds"],
        "p3_seeds": frozen["p3_seeds"],
        "screen_seed_participates": frozen["screen_seed_participates"],
        "freeze_timestamp": frozen["freeze_timestamp"],
        "read_timestamp": C.utcnow(),
        "git_commit": C.git_commit(),
        "data": {
            "source": "official PyG ZINC subset=True test",
            "n_test": 1000,
            "test_label_sha256": C.sha256_array(y_test_p3),
            "test_constructed_only_after_freeze": True,
            "historical_exposure": {
                "disclosure": "official ZINC test was exposed in earlier rounds; this is not blind",
                "historical_p3_test_mae_seed0_seed1": [0.069304, 0.070340],
                "source": "tracks/structural-sufficiency/results/phase2b/REPORT.md",
            },
        },
        "rows": rows,
        "summary": {
            "p3_valid": _stats(p3_valid),
            "p3_test": _stats(p3_test),
            "zsrt_valid": _stats(z_valid),
            "zsrt_test": _stats(z_test),
            "paired_valid_delta_zsrt_minus_p3": paired_valid,
            "paired_test_delta_zsrt_minus_p3": paired_test,
            "paired_valid_mean_delta": float(np.mean(list(paired_valid.values()))) if paired_valid else None,
            "paired_test_mean_delta": float(np.mean(list(paired_test.values()))) if paired_test else None,
        },
        "reading": [
            "best-valid checkpoints are the primary objects; no soup and no ensemble is used",
            "test was not used to select the config, epoch, seed or checkpoint",
            "P3 test is reported for the same best-valid objects; it is not a new blind baseline",
        ],
    }
    C.write_json(C.ANALYSIS / "test_scores.json", out)
    C.log(f"test summary: {out['summary']}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["score"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    score(args.device, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
