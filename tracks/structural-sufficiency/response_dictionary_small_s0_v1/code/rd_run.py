"""Training entry point for one (task, arm) unit.

    uv run python .../rd_run.py --task lipo --arm RESP --device cuda:0

Writes, under ``RD_RESULTS_DIR/<task>_<arm>``:

* ``curve.csv``               one row per epoch (Train/Val/Test raw RMSE+MAE)
* ``best.pt`` / ``final.pt``  best-Val and final state dicts
* ``predictions_best.npz`` / ``predictions_final.npz`` three splits, raw units
* ``summary.json``            selected epoch, numbers, hashes, timing
* ``status.json``             running/completed/failed + error text
* ``run.log``                 stdout mirror
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import rd_common as RC  # noqa: E402
import rd_model as RM  # noqa: E402
import rd_train as RT  # noqa: E402

TASK_DATASET = {"lipo": "ogbg-mollipo", "esol": "ogbg-molesol"}


def dataset_dims(dataset_name: str) -> dict:
    return RM.ogb_category_dims()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TASK_DATASET))
    parser.add_argument("--arm", required=True, choices=["RESP", "ADD"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None, help="override for smoke only")
    parser.add_argument("--init-state", default=None, help="path to a canonical init .pt")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = RC.load_config()
    run_commit = RC.git_commit()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    dataset_name = TASK_DATASET[args.task]
    out_dir = Path(args.out) if args.out else RC.run_dir(args.task, args.arm)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    log_handle = log_path.open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    status = {
        "task": args.task,
        "arm": args.arm,
        "dataset": dataset_name,
        "state": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": run_commit,
        "protocol_hash": RC.protocol_hash(config),
        "device": args.device,
        "argv": sys.argv,
    }
    RC.write_json(out_dir / "status.json", status)
    try:
        epochs = int(args.epochs if args.epochs is not None else config["epochs"])
        t0 = time.perf_counter()
        dataset = RT.build_dataset(dataset_name, args.task)
        log(
            f"data ready: {dataset.name} train/valid/test = "
            f"{len(dataset.splits['train'].row_id)}/{len(dataset.splits['valid'].row_id)}/"
            f"{len(dataset.splits['test'].row_id)} fragments {len(dataset.frag_data)} "
            f"cache {dataset.cache_sha256[:12]}"
        )
        model_config = RM.ModelConfig(
            hidden=int(config["hidden"]),
            latent=int(config["latent"]),
            n_layers=int(config["n_layers"]),
            dropout=float(config["dropout"]),
            mode=args.arm,
        )
        default_init = RC.DATA_DIR / "canonical_init_state.pt"
        expected_hash_path = RC.DATA_DIR / "canonical_init_state.sha256"
        if args.init_state:
            state = _load_state(Path(args.init_state))
        elif default_init.is_file():
            state = _load_state(default_init)
            if expected_hash_path.is_file():
                expected = expected_hash_path.read_text(encoding="utf-8").strip()
                if RM.state_hash(state) != expected:
                    raise RuntimeError("committed canonical init state hash mismatch")
        else:
            state = RM.canonical_init_state(dataset.dims, model_config, seed=int(config["seed"]))
        init_hash = RM.state_hash(state)
        init_path = out_dir / "init_state.pt"
        import torch

        torch.save({k: v.clone() for k, v in state.items()}, init_path)
        log(f"canonical init state {init_hash[:16]} -> {init_path.name}")
        order = RT.train_order(dataset, epochs, int(config["seed"]))
        order_path = out_dir / "train_order.npy"
        np.save(order_path, order)
        status.update(
            {
                "init_state_hash": init_hash,
                "init_state_sha256": RC.sha256_file(init_path),
                "train_order_sha256": RC.sha256_file(order_path),
                "data_cache_sha256": dataset.cache_sha256,
                "scaler": dataset.scaler,
                "n_parameters": RM.n_parameters(RM.ResponseModel(dataset.dims, model_config)),
                "epochs": epochs,
                "data_seconds": time.perf_counter() - t0,
            }
        )
        RC.write_json(out_dir / "status.json", status)

        result = RT.train_arm(
            dataset=dataset,
            arm=args.arm,
            config=config,
            device=args.device,
            init_state=state,
            order=order,
            epochs=epochs,
            log=log,
            eval_batch_size=int(config["eval_batch_size"]),
        )
        _save_outputs(out_dir, result, dataset, config, args.task, log, run_commit)
        status.update(
            {
                "state": "completed",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "seconds": result["seconds"],
                "best_epoch": result["best"]["epoch"],
                "best_valid_rmse_raw": result["best"]["valid_rmse_raw"],
                "n_parameters": result["n_parameters"],
            }
        )
        RC.write_json(out_dir / "status.json", status)
        log(f"completed in {result['seconds'] / 60:.1f} min; best epoch {result['best']['epoch']}")
        return 0
    except BaseException as exc:  # noqa: BLE001 - recorded, re-raised as exit code
        status.update(
            {
                "state": "failed",
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:],
            }
        )
        RC.write_json(out_dir / "status.json", status)
        log(f"FAILED {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        raise


def _load_state(path: Path) -> dict:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    return {k: v for k, v in state.items()}


def _save_outputs(out_dir: Path, result, dataset, config, task: str, log, run_commit: str) -> None:
    import torch

    curves = result["curves"]
    RC.write_csv(out_dir / "curve.csv", curves)
    torch.save(result["best_state"], out_dir / "best.pt")
    torch.save(result["final_state"], out_dir / "final.pt")
    for tag, pred in (("best", result["best_predictions"]), ("final", result["final_predictions"])):
        arrays = {}
        for split, payload in pred.items():
            for key, value in payload.items():
                arrays[f"{split}_{key}"] = np.asarray(value)
        arrays["selected_epoch"] = np.asarray([result["best"]["epoch"] if tag == "best" else len(curves)], dtype=np.int64)
        arrays["arm"] = np.asarray([result["arm"]])
        np.savez_compressed(out_dir / f"predictions_{tag}.npz", **arrays)
    last10 = [row for row in curves[-10:]]
    last30 = [row for row in curves[-30:]]
    prev30 = [row for row in curves[-60:-30]] if len(curves) >= 60 else []
    first30 = [row for row in curves[:30]]
    std = float(dataset.scaler["std"])
    summary = {
        "task": task,
        "dataset": dataset.name,
        "arm": result["arm"],
        "epochs": len(curves),
        "curves": curves,
        "best": result["best"],
        "best_metrics": _metrics_of(curves, result["best"]["epoch"]),
        "final_metrics": _metrics_of(curves, len(curves)),
        "last10_mean": {f"{split}_rmse_raw": float(np.mean([r[f"{split}_rmse_raw"] for r in last10])) for split in ("train", "valid", "test")},
        "last30_mean": {f"{split}_rmse_raw": float(np.mean([r[f"{split}_rmse_raw"] for r in last30])) for split in ("train", "valid", "test")},
        "prev30_mean": ({f"{split}_rmse_raw": float(np.mean([r[f"{split}_rmse_raw"] for r in prev30])) for split in ("train", "valid", "test")} if prev30 else None),
        "first30_mean": {f"{split}_rmse_raw": float(np.mean([r[f"{split}_rmse_raw"] for r in first30])) for split in ("train", "valid", "test")},
        "target_scaler": dataset.scaler,
        "target_std": std,
        "n_parameters": result["n_parameters"],
        "init_state_hash": result["init_state_hash"],
        "loaded_state_hash": result["loaded_state_hash"],
        "steps_per_epoch": result["steps_per_epoch"],
        "global_step": result["global_step"],
        "seconds": result["seconds"],
        "eval_batch_size": result["eval_batch_size"],
        "precision": {
            "dtype": "float32",
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "cudnn_benchmark": False,
            "float32_matmul_precision": "highest",
        },
        "data_cache_sha256": dataset.cache_sha256,
        "protocol_hash": RC.protocol_hash(config),
        "commit": run_commit,
        "environment": RC.environment_snapshot(),
        "artifact_sha256": {},
    }
    # Horizon check (frozen rule, no continuation).
    late = summary["last30_mean"]["valid_rmse_raw"]
    early = summary["prev30_mean"]["valid_rmse_raw"] if summary["prev30_mean"] else None
    best_last30 = min(last30, key=lambda r: r["valid_rmse_raw"])
    best_last10 = min(last10, key=lambda r: r["valid_rmse_raw"])
    delta = (late - early) if early is not None else None
    summary["horizon"] = {
        "comparison": "mean Val RMSE epochs 331-360 minus epochs 301-330" if early is not None else "not enough epochs",
        "delta_last30_minus_prev30_valid": delta,
        "threshold": -float(config["target_std_criterion"]) * std,
        "best_epoch_in_last30": int(best_last30["epoch"]),
        "best_epoch_in_last10": int(best_last10["epoch"]),
        "horizon_limited": bool(
            delta is not None
            and delta <= -float(config["target_std_criterion"]) * std
            and int(best_last10["epoch"]) >= len(curves) - 9
        ),
    }
    # Split dispersion of view scores + fallback shares (fresh eval on best state).
    import rd_train as RT

    model = RM.build_model(dataset.dims, RM.ModelConfig(
        hidden=int(config["hidden"]), latent=int(config["latent"]), n_layers=int(config["n_layers"]),
        dropout=float(config["dropout"]), mode=result["arm"]), result["best_state"]).to("cpu")
    diag = {}
    for split_name in ("train", "valid", "test"):
        split = dataset.splits[split_name]
        n = len(split.row_id)
        model.eval()
        sum_sq = 0.0
        n_views_total = 0
        n_mol_with_views = 0
        fb = np.zeros(n, dtype=bool)
        with torch.no_grad():
            for start in range(0, n, 256):
                idx = np.arange(start, min(n, start + 256))
                mean_score, extras = RT.molecule_scores(model, dataset, split, idx, "cpu")
                view_mol = np.repeat(np.arange(idx.size), extras["n_views_per_mol"])
                view_scores = extras["view_score_std"]
                if view_scores.size:
                    dev = view_scores - mean_score[view_mol]
                    sum_sq += float(np.sum(dev ** 2))
                    n_views_total += int(dev.size)
                    n_mol_with_views += int(np.sum(extras["n_views_per_mol"] > 0))
                fb[idx] = extras["fallback"]
        dispersion = float(np.sqrt(sum_sq / n_views_total)) if n_views_total else 0.0
        diag[split_name] = {
            "view_dispersion_raw": dispersion * std,
            "view_dispersion_std": dispersion,
            "fallback_molecule_fraction": float(fb.mean()),
            "n_molecules": int(n),
            "n_molecules_with_views": int(n_mol_with_views),
            "n_views": int(n_views_total),
            "view_fraction_molecules_with_cuts": float(1.0 - fb.mean()),
            "eval_batch_reductions": int(RT._EVAL_BATCH_STATE.get("reductions", 0)),
            "effective_eval_batch": int(RT._EVAL_BATCH_STATE.get("effective", 0)),
        }
    summary["diagnostics"] = diag
    summary["horizon_limited_candidate"] = summary["horizon"]["horizon_limited"]
    RC.write_json(out_dir / "summary.json", summary)
    log(
        "best epoch %d: train %.4f valid %.4f test %.4f (raw), fallback frac train %.3f valid %.3f test %.3f"
        % (
            result["best"]["epoch"],
            summary["best_metrics"]["train_rmse_raw"],
            summary["best_metrics"]["valid_rmse_raw"],
            summary["best_metrics"]["test_rmse_raw"],
            diag["train"]["fallback_molecule_fraction"],
            diag["valid"]["fallback_molecule_fraction"],
            diag["test"]["fallback_molecule_fraction"],
        )
    )


def _metrics_of(curves, epoch: int) -> dict:
    row = curves[epoch - 1]
    return {
        key: row[key]
        for key in (
            "epoch",
            "train_rmse_raw",
            "valid_rmse_raw",
            "test_rmse_raw",
            "train_mae_raw",
            "valid_mae_raw",
            "test_mae_raw",
            "train_loss",
        )
    }


if __name__ == "__main__":
    raise SystemExit(main())
