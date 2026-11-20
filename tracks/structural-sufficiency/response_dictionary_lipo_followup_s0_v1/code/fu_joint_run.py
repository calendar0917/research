"""Formal unit of this round: Lipo JOINT, seed 0, 360 epochs.

    uv run python code/fu_joint_run.py --device cuda:0

One unit only.  It loads the committed S0 canonical initial state, runs the
targeted pre-flight checks, then the frozen S0 recipe with the JOINT fusion, and
writes the same artefact set as the parent round plus ``preflight.json``.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import fu_checks as FK  # noqa: E402
import fu_common as FC  # noqa: E402
import fu_model as FM  # noqa: E402
import fu_train as FT  # noqa: E402
import rd_common as RC  # noqa: E402
import rd_model as RM  # noqa: E402
import rd_train as RT  # noqa: E402

DATASET = "ogbg-mollipo"
TASK = "lipo"
ARM = "JOINT"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None, help="override for smoke only")
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args(argv)

    recipe = FC.load_parent_config()
    followup = FC.load_config()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")

    out_dir = Path(args.out) if args.out else FC.run_dir(TASK, ARM)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (out_dir / "run.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    status = {
        "task": TASK,
        "arm": ARM,
        "dataset": DATASET,
        "state": "running",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "commit": RC.git_commit(),
        "protocol_hash": FC.protocol_hash(),
        "parent_protocol_hash": RC.protocol_hash(recipe),
        "device": args.device,
        "argv": sys.argv,
    }
    RC.write_json(out_dir / "status.json", status)
    try:
        epochs = int(args.epochs if args.epochs is not None else recipe["epochs"])
        t0 = time.perf_counter()
        dataset = RT.build_dataset(DATASET, TASK)
        log(
            f"data ready: {dataset.name} train/valid/test = "
            f"{len(dataset.splits['train'].row_id)}/{len(dataset.splits['valid'].row_id)}/"
            f"{len(dataset.splits['test'].row_id)} fragments {len(dataset.frag_data)} "
            f"cache {dataset.cache_sha256[:12]}"
        )

        init_path = FC.S0_DIR / "data" / "canonical_init_state.pt"
        state = FM.load_canonical_state(init_path, FC.EXPECTED_INIT_STATE_HASH)
        saved_init = out_dir / "init_state.pt"
        torch.save({k: v.clone() for k, v in state.items()}, saved_init)
        log(f"canonical S0 init state {RM.state_hash(state)[:16]} <- {init_path}")

        order = RT.train_order(dataset, epochs, int(recipe["seed"]))
        order_path = out_dir / "train_order.npy"
        np.save(order_path, order)
        order_sha = RC.sha256_file(order_path)

        preflight = {"state": "skipped"}
        if not args.skip_preflight:
            preflight = FK.run_preflight(
                dataset=dataset,
                config=recipe,
                state=state,
                order=order,
                order_sha256=order_sha,
                device=args.device,
                parent_curve=FC.S0_RESULTS / "lipo_RESP" / "curve.csv",
                log=log,
            )
            RC.write_json(out_dir / "preflight.json", preflight)
            log("preflight complete")

        status.update(
            {
                "init_state_hash": RM.state_hash(state),
                "init_state_sha256": RC.sha256_file(saved_init),
                "train_order_sha256": order_sha,
                "data_cache_sha256": dataset.cache_sha256,
                "scaler": dataset.scaler,
                "n_parameters": RM.n_parameters(FM.build_model(dataset.dims, FM.model_config(recipe, ARM), state)),
                "epochs": epochs,
                "data_seconds": time.perf_counter() - t0,
                "preflight_ok": bool(preflight.get("ok", False)) if preflight.get("state") != "skipped" else None,
            }
        )
        RC.write_json(out_dir / "status.json", status)

        result = FT.train_arm(
            dataset=dataset,
            arm=ARM,
            config=recipe,
            device=args.device,
            init_state=state,
            order=order,
            epochs=epochs,
            log=log,
            eval_batch_size=int(recipe["eval_batch_size"]),
            model_factory=FM.build_model,
        )
        _save_outputs(out_dir, result, dataset, recipe, followup, log, status["commit"])
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


def _save_outputs(out_dir: Path, result, dataset, recipe, followup, log, run_commit: str) -> None:
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
    last10 = list(curves[-10:])
    last30 = list(curves[-30:])
    prev30 = list(curves[-60:-30]) if len(curves) >= 60 else []
    first30 = list(curves[:30])
    std = float(dataset.scaler["std"])
    summary = {
        "task": TASK,
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
        "protocol_hash": FC.protocol_hash(),
        "parent_protocol_hash": RC.protocol_hash(recipe),
        "commit": run_commit,
        "followup_config": followup,
        "environment": RC.environment_snapshot(),
        "artifact_sha256": {},
    }
    late = summary["last30_mean"]["valid_rmse_raw"]
    early = summary["prev30_mean"]["valid_rmse_raw"] if summary["prev30_mean"] else None
    best_last30 = min(last30, key=lambda r: r["valid_rmse_raw"])
    best_last10 = min(last10, key=lambda r: r["valid_rmse_raw"])
    delta = (late - early) if early is not None else None
    summary["horizon"] = {
        "comparison": "mean Val RMSE epochs 331-360 minus epochs 301-330" if early is not None else "not enough epochs",
        "delta_last30_minus_prev30_valid": delta,
        "threshold": -float(recipe["target_std_criterion"]) * std,
        "best_epoch_in_last30": int(best_last30["epoch"]),
        "best_epoch_in_last10": int(best_last10["epoch"]),
        "horizon_limited": bool(
            delta is not None
            and delta <= -float(recipe["target_std_criterion"]) * std
            and int(best_last10["epoch"]) >= len(curves) - 9
        ),
    }
    model = FM.build_model(
        dataset.dims, FM.model_config(recipe, ARM), result["best_state"]
    ).to("cpu")
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
        "best epoch %d: train %.4f valid %.4f test %.4f (raw), view dispersion val %.4f, fallback frac train %.3f valid %.3f test %.3f"
        % (
            result["best"]["epoch"],
            summary["best_metrics"]["train_rmse_raw"],
            summary["best_metrics"]["valid_rmse_raw"],
            summary["best_metrics"]["test_rmse_raw"],
            diag["valid"]["view_dispersion_raw"],
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
