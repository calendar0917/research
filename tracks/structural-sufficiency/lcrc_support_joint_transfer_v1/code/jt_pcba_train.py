"""Formal 25-epoch prefix training of ``W2_JOINT`` and ``FIXED1_JOINT``.

The data, batched order plan, optimiser schedule, masked BCE, full-split
evaluation and official OGB macro AP are inherited from
``direct_structural_computation_recurrence_v0`` (itself read-only reuse of the
W2 scratch recipe).  The horizon is the 25-epoch **prefix** of the historical
40-epoch cosine schedule; nothing is recompressed.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch  # noqa: E402

import jt_common as C  # noqa: E402
import jt_models as M  # noqa: E402
import jt_support as S  # noqa: E402

S._prepare()
M.prepare_imports()

import ds_common as DC  # noqa: E402
import ds_train as DT  # noqa: E402
import lr_data as LD  # noqa: E402
import pc_model as PCM  # noqa: E402

CURVE_FIELDS = [
    "epoch",
    "global_step",
    "steps_per_epoch",
    "learning_rate",
    "train_bce",
    "train_ap",
    "valid_bce",
    "valid_ap",
    "test_bce",
    "test_ap",
    "n_tasks_scored_train",
    "n_tasks_scored_valid",
    "n_tasks_scored_test",
    "epoch_train_seconds",
    "epoch_eval_seconds",
    "epoch_seconds",
    "cumulative_seconds",
    "step_ms_mean",
    "best_epoch_so_far",
    "best_valid_ap_so_far",
    "is_best",
]
PER_TASK_FIELDS = ["epoch", "split", "task", "ap", "observed", "positive", "negative"]


def _atomic_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row[k]) for k in fields})


def _checkpoint(
    path: Path,
    *,
    arm: str,
    epoch: int,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    plan_meta: Mapping[str, Any],
    model_info: Mapping[str, Any],
    support: Any,
    metrics: Mapping[str, Any],
    kind: str,
) -> None:
    payload = {
        "schema": "lcrc-support-joint-transfer-checkpoint-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "family": "pcba",
        "arm": arm,
        "seed": C.SEED,
        "horizon": int(C.PCBA_RECIPE["horizon"]),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "kind": str(kind),
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "batch_plan": dict(plan_meta),
        "model_info": dict(model_info),
        "support": {
            "n_omega": int(support.n_omega),
            "rank_X": int(support.rank),
            "omega_code_sha256": C.sha256_array(np.asarray(support.omega_code, dtype=np.int64)),
            "P_sha256": C.sha256_array(np.asarray(support.P, dtype=np.float64)),
        },
        "metrics": dict(metrics),
        "test_role": C.PCBA_TEST_ROLE,
        "test_used_for_selection": False,
    }
    _atomic_save(payload, path)


def run(arm: str, out_dir: Path, *, epochs: int, device: torch.device, hard_seconds: float) -> dict[str, Any]:
    if arm not in C.PCBA_ARMS:
        raise KeyError(arm)
    if int(epochs) != int(C.PCBA_RECIPE["horizon"]):
        raise RuntimeError("the frozen PCBA horizon is exactly 25 epochs (40-epoch schedule prefix)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    started_utc = C.utcnow()

    task = LD.load_task()
    support = S.load_support(C.ANALYSIS / "support/pcba_support.npz")
    if support.n_omega != 214:
        raise RuntimeError(f"unexpected PCBA Omega size {support.n_omega}")
    model, info = M.build_pcba_model(task, support, arm)
    model = model.to(device)
    optimizer, opt_audit = DT.build_optimizer(model)
    plan, plan_audit = DC.plan_for_seed(task, C.SEED)
    test_info = DC.load_test_labels(task, cache_dir=C.RUNS / "_shared")
    test_labels = np.asarray(test_info["labels"], dtype=np.float32)
    if test_labels.shape != (int(task.test_rows.size), int(C.PCBA_N_TASKS)):
        raise RuntimeError("official Test label matrix shape mismatch")
    plan_meta = DC.plan_meta(plan, int(epochs), C.SEED)
    if plan_meta["prefix_sha256"] != C.W2_EXPECTED["prefix25_sha256"]:
        raise RuntimeError(f"25-epoch plan prefix {plan_meta['prefix_sha256']} != frozen")

    DC.seed_everything(C.SEED)
    rng_start = DC.rng_state_hashes(DC.capture_rng_state())
    stager = DT.PinnedStager(device, depth=2)
    clip = float(DC.RECIPE["gradient_clip"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    curve_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    best_epoch: int | None = None
    best_valid_ap = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    final_state: dict[str, torch.Tensor] | None = None
    global_step = 0
    watch = DT.Watchdog(float(hard_seconds))
    stop_reason = "COMPLETED"

    def eval_all(epoch: int) -> dict[str, Any]:
        before = DC.rng_state_hashes(DC.capture_rng_state())["combined_sha256"]
        model.eval()
        with torch.inference_mode():
            reports = {
                "train": DC.evaluate_split(model, task, device, "train", per_task=True, reason=f"epoch_{epoch}"),
                "valid": DC.evaluate_split(model, task, device, "valid", per_task=True, reason=f"epoch_{epoch}"),
                "test": DC.evaluate_split(
                    model, task, device, "test", y=test_labels, per_task=True, reason=f"epoch_{epoch}"
                ),
            }
        after = DC.rng_state_hashes(DC.capture_rng_state())["combined_sha256"]
        if before != after:
            raise RuntimeError("full-split evaluation moved the training RNG state")
        model.train()
        return reports

    for epoch in range(1, int(epochs) + 1):
        if watch.stop.is_set():
            stop_reason = "HARD_WALL_LIMIT"
            break
        model.train()
        t_epoch = time.perf_counter()
        batches = plan[epoch - 1]["batches"]
        losses: list[float] = []
        step_ms: list[float] = []
        for rows in batches:
            t0 = time.perf_counter()
            batch = stager.batch(PCM.collate(task.cache, rows.tolist()))
            y = stager.labels(task.labels[rows])
            out = DT.train_step(
                model,
                optimizer,
                batch,
                y,
                global_step=global_step,
                clip=clip,
                audit_new=False,
            )
            global_step += 1
            losses.append(float(out["loss"]))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_ms.append(float(time.perf_counter() - t0) * 1e3)
        train_seconds = float(time.perf_counter() - t_epoch)

        reports = eval_all(epoch)
        eval_seconds = float(sum(reports[s]["seconds"] for s in ("train", "valid", "test")))
        valid_ap = float(reports["valid"]["ap"])
        strict = best_epoch is None or valid_ap > float(best_valid_ap)
        if strict:
            best_epoch = int(epoch)
            best_valid_ap = valid_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "epoch": int(epoch),
                "valid": {k: reports["valid"][k] for k in ("ap", "bce", "n_tasks_scored")},
                "test": {k: reports["test"][k] for k in ("ap", "bce", "n_tasks_scored")},
                "train": {k: reports["train"][k] for k in ("ap", "bce", "n_tasks_scored")},
            }
            _checkpoint(
                out_dir / "checkpoint_best.pt",
                arm=arm,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                plan_meta=DC.plan_meta(plan, int(epoch), C.SEED),
                model_info=info,
                support=support,
                metrics=best_metrics,
                kind="best_valid_ap",
            )
        for split in ("train", "valid", "test"):
            rep = reports[split]
            for t, ap in enumerate(rep["per_task_ap"]):
                per_task_rows.append(
                    {
                        "epoch": int(epoch),
                        "split": split,
                        "task": int(t),
                        "ap": "" if ap is None else float(ap),
                        "observed": int(rep["observed"][t]),
                        "positive": int(rep["positive"][t]),
                        "negative": int(rep["negative"][t]),
                    }
                )
        final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        arr = np.asarray(step_ms, dtype=np.float64)
        row = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "steps_per_epoch": int(len(batches)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_bce": float(reports["train"]["bce"]),
            "train_ap": float(reports["train"]["ap"]),
            "valid_bce": float(reports["valid"]["bce"]),
            "valid_ap": valid_ap,
            "test_bce": float(reports["test"]["bce"]),
            "test_ap": float(reports["test"]["ap"]),
            "n_tasks_scored_train": int(reports["train"]["n_tasks_scored"]),
            "n_tasks_scored_valid": int(reports["valid"]["n_tasks_scored"]),
            "n_tasks_scored_test": int(reports["test"]["n_tasks_scored"]),
            "epoch_train_seconds": train_seconds,
            "epoch_eval_seconds": eval_seconds,
            "epoch_seconds": float(time.perf_counter() - t_epoch),
            "cumulative_seconds": float(time.perf_counter() - t_start),
            "step_ms_mean": float(arr.mean()) if arr.size else None,
            "best_epoch_so_far": int(best_epoch) if best_epoch is not None else None,
            "best_valid_ap_so_far": float(best_valid_ap),
            "is_best": bool(strict),
        }
        curve_rows.append(row)
        _write_csv(out_dir / "train_curve.csv", curve_rows, CURVE_FIELDS)
        _write_csv(out_dir / "per_task_ap.csv", per_task_rows, PER_TASK_FIELDS)
        print(
            f"[{arm}] ep {epoch}/{epochs} trainAP {row['train_ap']:.6f} validAP {valid_ap:.6f} "
            f"testAP {row['test_ap']:.6f} lr {row['learning_rate']:.3e} train {train_seconds:.1f}s "
            f"eval {eval_seconds:.1f}s {'*best*' if strict else ''}",
            flush=True,
        )

    watch.cancel()
    stager.release()
    if best_state is None:
        raise RuntimeError("no selectable epoch completed")
    _checkpoint(
        out_dir / f"checkpoint_epoch{epochs}.pt",
        arm=arm,
        epoch=len(curve_rows),
        global_step=global_step,
        model=_StateModel(final_state),
        optimizer=optimizer,
        plan_meta=plan_meta,
        model_info=info,
        support=support,
        metrics={"epoch": len(curve_rows), "final": {k: curve_rows[-1][k] for k in ("valid_ap", "test_ap")}},
        kind="horizon_endpoint",
    )
    _checkpoint(
        out_dir / "checkpoint_last.pt",
        arm=arm,
        epoch=len(curve_rows),
        global_step=global_step,
        model=_StateModel(final_state),
        optimizer=optimizer,
        plan_meta=plan_meta,
        model_info=info,
        support=support,
        metrics={"epoch": len(curve_rows), "final": {k: curve_rows[-1][k] for k in ("valid_ap", "test_ap")}},
        kind="last",
    )

    # replay the selected epoch to store logits / per-task AP
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=True)
    best_eval = {}
    model.eval()
    with torch.inference_mode():
        for split in ("train", "valid", "test"):
            best_eval[split] = DC.evaluate_split(
                model,
                task,
                device,
                split,
                y=test_labels if split == "test" else None,
                per_task=True,
                keep_logits=True,
                reason="best_checkpoint_replay",
            )
    np.save(out_dir / "valid_logits_best.npy", best_eval["valid"]["logits"].astype(np.float32))
    np.save(out_dir / "test_logits_best.npy", best_eval["test"]["logits"].astype(np.float32))
    C.write_json(
        out_dir / "best_checkpoint_eval.json",
        {
            "epoch": int(best_epoch),
            "selection_metric": "valid.ap",
            "valid": {k: best_eval["valid"][k] for k in ("ap", "bce", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "test": {k: best_eval["test"][k] for k in ("ap", "bce", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "train": {k: best_eval["train"][k] for k in ("ap", "bce", "n_tasks_scored", "per_task_ap", "observed", "positive", "negative")},
            "test_role": C.PCBA_TEST_ROLE,
            "test_used_for_selection": False,
        },
    )
    # final-epoch logits
    model.load_state_dict({k: v.to(device) for k, v in final_state.items()}, strict=True)
    model.eval()
    with torch.inference_mode():
        final_valid = DC.evaluate_split(model, task, device, "valid", per_task=True, keep_logits=True, reason="final_replay")
        final_test = DC.evaluate_split(model, task, device, "test", y=test_labels, per_task=True, keep_logits=True, reason="final_replay")
    np.save(out_dir / "valid_logits_final.npy", final_valid["logits"].astype(np.float32))
    np.save(out_dir / "test_logits_final.npy", final_test["logits"].astype(np.float32))

    last20 = {
        f"mean_last20_{split}_ap": float(np.mean([r[f"{split}_ap"] for r in curve_rows[-20:]]))
        for split in ("train", "valid", "test")
    }
    summary = {
        "schema": "lcrc-support-joint-transfer-run-v1",
        "round_id": C.ROUND_ID,
        "family": "pcba",
        "dataset": "ogbg-molpcba",
        "arm": arm,
        "seed": C.SEED,
        "horizon": int(epochs),
        "status": "COMPLETED" if stop_reason == "COMPLETED" else stop_reason,
        "recipe": dict(C.PCBA_RECIPE),
        "widths": dict(DC.BC.config_for_scale(2)),
        "batch_plan": plan_meta,
        "plan_audit": plan_audit,
        "parameters": int(info["parameters"]),
        "base_parameters": int(info["base_parameters"]),
        "joint_module_parameters": int(info["joint_module_parameters"]),
        "train_supported_joint_parameters": int(info["train_supported_joint_parameters"]),
        "model_info": info,
        "support": S.omega_identity(support),
        "support_summary": support.summary(),
        "optimizer_audit": opt_audit,
        "test_label_read": {k: v for k, v in test_info.items() if k != "labels"},
        "test_role": C.PCBA_TEST_ROLE,
        "test_used_for_selection": False,
        "curve": curve_rows,
        "best_epoch": int(best_epoch),
        "best_valid_ap": float(best_valid_ap),
        "best_checkpoint_eval": {
            "valid_ap": float(best_eval["valid"]["ap"]),
            "test_ap": float(best_eval["test"]["ap"]),
            "valid_bce": float(best_eval["valid"]["bce"]),
            "test_bce": float(best_eval["test"]["bce"]),
        },
        "final": {"epoch": len(curve_rows), "valid_ap": float(curve_rows[-1]["valid_ap"]), "test_ap": float(curve_rows[-1]["test_ap"])},
        "last20": last20,
        "rng_start": rng_start,
        "train_seconds": float(sum(r["epoch_train_seconds"] for r in curve_rows)),
        "eval_seconds": float(sum(r["epoch_eval_seconds"] for r in curve_rows)),
        "wall_seconds": float(time.perf_counter() - t_start),
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "checkpoint_hashes": {
            "checkpoint_best.pt": C.sha256_file(out_dir / "checkpoint_best.pt"),
            f"checkpoint_epoch{epochs}.pt": C.sha256_file(out_dir / f"checkpoint_epoch{epochs}.pt"),
            "checkpoint_last.pt": C.sha256_file(out_dir / "checkpoint_last.pt"),
        },
        "environment": C.environment(),
        "device": str(device),
        "started_utc": started_utc,
        "finished_utc": C.utcnow(),
    }
    C.write_json(out_dir / "run.json", summary)
    return summary


class _StateModel(torch.nn.Module):
    def __init__(self, state: Mapping[str, torch.Tensor] | None):
        super().__init__()
        self._state = state or {}

    def state_dict(self, *args, **kwargs):  # noqa: D401
        return self._state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=C.PCBA_ARMS)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=int(C.PCBA_RECIPE["horizon"]))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hard-seconds", type=float, default=10800.0)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    try:
        summary = run(args.arm, Path(args.out_dir), epochs=args.epochs, device=device, hard_seconds=args.hard_seconds)
        print(
            f"COMPLETED {args.arm} best_epoch={summary['best_epoch']} "
            f"validAP={summary['best_valid_ap']:.9f} testAP={summary['best_checkpoint_eval']['test_ap']:.9f}"
        )
    except Exception:
        traceback.print_exc()
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        C.write_json(out / "run.failed.json", {"status": "FAILED", "traceback": traceback.format_exc(), "utc": C.utcnow()})
        raise


if __name__ == "__main__":
    main()
