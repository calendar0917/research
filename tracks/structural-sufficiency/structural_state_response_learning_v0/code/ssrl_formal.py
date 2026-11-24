"""Launch the three SSRL v0 arms on two GPUs and record the budget.

Formal schedule: ABS and PAIR run concurrently on GPU 0 / GPU 1, then RECT runs
on GPU 0.  The horizon is read from the frozen config (5 or 3); this launcher
never changes it based on a prediction or a response score.

Preflight: one real epoch of ABS plus full official-Valid inference, followed
by a timing projection only.  The horizon decision is made from wall-clock
measurements, before the formal run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ssrl_common as C


ARMS = ("ABS", "PAIR", "RECT")


def _run(arm: str, gpu: int, epochs: int, out_dir: Path, max_seconds: float,
         *, preflight: bool, max_steps: int | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    cmd = [
        sys.executable, str(CODE / "ssrl_train.py"),
        "--arm", arm, "--epochs", str(epochs),
        "--device", "cuda", "--out-dir", str(out_dir),
        "--max-seconds", str(float(max_seconds)),
    ]
    if preflight:
        cmd += ["--preflight"]
        if max_steps is not None:
            cmd += ["--max-steps", str(int(max_steps))]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    env["OMP_NUM_THREADS"] = "8"
    env["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=str(C.REPO_ROOT), env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
        code = proc.wait()
    finished = time.time()
    seconds = finished - started
    run_json = out_dir / "run.json"
    run = C.read_json(run_json) if run_json.exists() else None
    return {
        "arm": arm,
        "gpu": int(gpu),
        "command": cmd,
        "exit_code": int(code),
        "started_unix": started,
        "finished_unix": finished,
        "wall_seconds": float(seconds),
        "run_json": C.track_rel(run_json) if run_json.exists() else None,
        "log": C.track_rel(log_path),
        "run": run,
        "ok": bool(code == 0 and run is not None and run.get("status") == "OK"),
    }


def _summary(record: dict) -> dict:
    run = record.get("run") or {}
    return {
        "arm": record["arm"],
        "gpu": record["gpu"],
        "exit_code": record["exit_code"],
        "ok": record["ok"],
        "wall_seconds": record.get("wall_seconds"),
        "run_seconds": run.get("seconds"),
        "gpu_hours_process": run.get("gpu_hours_process"),
        "best_epoch": run.get("best_epoch"),
        "best_valid_mae_eV": run.get("best_valid_mae_eV"),
        "final_epoch": run.get("final_epoch"),
        "final_valid_mae_eV": run.get("final_valid_mae_eV"),
        "steps_per_epoch": run.get("steps_per_epoch"),
        "epochs": run.get("epochs"),
        "log": record["log"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight", "formal"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=880.0)
    args = parser.parse_args(argv)

    cfg = C.read_json(C.CONFIG_PATH)
    manifest = C.read_json(C.DATA_DIR / "train_plan_manifest.json")
    epochs = int(args.epochs if args.epochs is not None else cfg["horizon_epochs"])
    if epochs not in (3, 5):
        raise RuntimeError("horizon must be 3 or 5")
    if args.mode == "formal" and epochs != int(cfg["horizon_epochs"]):
        raise RuntimeError(
            f"formal horizon {epochs} != frozen config horizon {cfg['horizon_epochs']}; "
            "change the config deliberately and commit it before the formal run"
        )
    steps_per_epoch = int(manifest["steps_per_epoch"])
    t0 = time.time()

    if args.mode == "preflight":
        print(f"preflight: one epoch ({steps_per_epoch} steps), full Valid inference", flush=True)
        rec = _run("ABS", 0, epochs, C.RUNS_DIR / "preflight_ABS", args.max_seconds,
                   preflight=True, max_steps=steps_per_epoch)
        wall = time.time() - t0
        arms = [_summary(rec)]
        run = rec.get("run") or {}
        curve = run.get("curve") or []
        if len(curve) >= 2:
            epoch_train = float(curve[1].get("epoch_train_seconds") or 0.0)
            epoch_eval = float(curve[1].get("valid_seconds") or 0.0) + float(
                curve[1].get("train_subset_seconds") or 0.0)
            epoch0_eval = float(curve[0].get("valid_seconds") or 0.0) + float(
                curve[0].get("train_subset_seconds") or 0.0)
            overhead = max(float(run.get("seconds") or rec["wall_seconds"])
                           - epoch0_eval - epoch_train - epoch_eval, 0.0)
            per_arm = overhead + epoch0_eval + float(epochs) * (epoch_train + epoch_eval)
            per_arm_3 = overhead + epoch0_eval + 3.0 * (epoch_train + epoch_eval)
        else:
            per_arm = float(rec["wall_seconds"]) * (float(epochs) + 1.0) / 2.0
            per_arm_3 = per_arm * 3.0 / float(epochs)
        estimate_wall = 2.0 * per_arm
        estimate_gpu_hours = 3.0 * per_arm / 3600.0
        recommendation = 5 if estimate_wall <= 600.0 else 3
        payload = {
            "schema": "ssrl-v0-preflight-timing",
            "created_utc": C.now_utc(),
            "git_commit": C.git_commit(),
            "git_dirty": C.git_dirty(),
            "mode": "preflight",
            "epochs_in_config": epochs,
            "steps_per_epoch": steps_per_epoch,
            "measured": arms,
            "estimate_per_arm_seconds": per_arm,
            "estimate_formal_wall_seconds": estimate_wall,
            "estimate_formal_gpu_hours": estimate_gpu_hours,
            "estimate_formal_wall_seconds_if_3_epochs": 2.0 * per_arm_3,
            "target_wall_seconds": cfg["budget"]["formal_wall_target_seconds"],
            "hard_ceiling_seconds": cfg["budget"]["formal_wall_hard_ceiling_seconds"],
            "recommended_horizon_if_timing_only": recommendation,
            "rule": "5 epochs if the two-wave projection is <= 600 s, else 3; timing only",
            "wall_seconds": wall,
        }
        C.write_json(C.ANALYSIS_DIR / "preflight_timing.json", payload)
        print(json.dumps(payload, indent=2)[:4000], flush=True)
        return 0 if rec["ok"] else 1

    print(f"formal: {epochs} epochs x {steps_per_epoch} steps, arms ABS/PAIR then RECT", flush=True)
    for arm in ("ABS", "PAIR"):
        (C.RUNS_DIR / f"PCQM_SSRL_{arm}").mkdir(parents=True, exist_ok=True)
    # first wave: ABS on GPU0, PAIR on GPU1
    p_abs = subprocess.Popen(
        [sys.executable, str(CODE / "ssrl_train.py"), "--arm", "ABS",
         "--epochs", str(epochs), "--device", "cuda",
         "--out-dir", str(C.RUNS_DIR / "PCQM_SSRL_ABS"),
         "--max-seconds", str(args.max_seconds)],
        cwd=str(C.REPO_ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "8",
             "PYTHONUNBUFFERED": "1"},
        stdout=(C.RUNS_DIR / "PCQM_SSRL_ABS" / "train.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    p_pair = subprocess.Popen(
        [sys.executable, str(CODE / "ssrl_train.py"), "--arm", "PAIR",
         "--epochs", str(epochs), "--device", "cuda",
         "--out-dir", str(C.RUNS_DIR / "PCQM_SSRL_PAIR"),
         "--max-seconds", str(args.max_seconds)],
        cwd=str(C.REPO_ROOT),
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "8",
             "PYTHONUNBUFFERED": "1"},
        stdout=(C.RUNS_DIR / "PCQM_SSRL_PAIR" / "train.log").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    wave1_t0 = time.time()
    code_abs = p_abs.wait()
    code_pair = p_pair.wait()
    wave1 = time.time() - wave1_t0
    # second wave: RECT on GPU0
    rec_rect = _run("RECT", 0, epochs, C.RUNS_DIR / "PCQM_SSRL_RECT",
                    args.max_seconds, preflight=False, max_steps=None)
    total_wall = time.time() - t0
    records = []
    for arm, code, gpu in (("ABS", code_abs, 0), ("PAIR", code_pair, 1)):
        run_path = C.RUNS_DIR / f"PCQM_SSRL_{arm}" / "run.json"
        run = C.read_json(run_path) if run_path.exists() else None
        records.append({
            "arm": arm, "gpu": gpu, "exit_code": int(code),
            "run_json": C.track_rel(run_path), "run": run,
            "log": C.track_rel(C.RUNS_DIR / f"PCQM_SSRL_{arm}" / "train.log"),
            "ok": bool(code == 0 and run is not None and run.get("status") == "OK"),
        })
    records.append(rec_rect)
    gpu_hours = sum(float((r.get("run") or {}).get("gpu_hours_process") or 0.0)
                    for r in records)
    payload = {
        "schema": "ssrl-v0-formal-timing",
        "created_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "mode": "formal",
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "wave1_wall_seconds": wave1,
        "total_wall_seconds": total_wall,
        "gpu_hours_process_sum": gpu_hours,
        "arms": [_summary(r) for r in records],
        "within_hard_ceiling": bool(total_wall <= float(cfg["budget"]["formal_wall_hard_ceiling_seconds"])),
        "within_wall_target": bool(300.0 <= total_wall <= 600.0),
        "within_gpu_budget": bool(gpu_hours <= float(cfg["budget"]["gpu_hours_hard_ceiling"])),
        "all_arms_ok": bool(all(r["ok"] for r in records)),
        "budget": cfg["budget"],
    }
    C.write_json(C.ANALYSIS_DIR / "formal_timing.json", payload)
    print(json.dumps(payload, indent=2)[:5000], flush=True)
    return 0 if payload["all_arms_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
