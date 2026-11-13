"""Sequential run queue for one GPU.

One ``rr`` job per GPU runs a list of ``candidate:dataset[:recipe]`` units in
order.  Completed units (a ``train.json`` with ``status == completed``) are
skipped, so an interrupted queue resumes without duplicating finished jobs.

Every unit's wall time and charged GPU hours are appended to
``analysis/queue_<purpose>.jsonl``; the queue never sleeps and never retries a
scientific failure (only infrastructure errors are re-raised).
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import torch

import lcpr_common as C


def parse_spec(spec: str) -> dict[str, Any]:
    parts = spec.split(":")
    if len(parts) < 2:
        raise SystemExit(f"bad job spec {spec!r}; expected candidate:dataset[:recipe[:epochs[:smoke]]]")
    out = {"candidate": parts[0], "dataset": parts[1]}
    out["recipe"] = parts[2] if len(parts) > 2 and parts[2] else None
    digits = [p for p in parts[3:] if p and p.isdigit()]
    out["epochs"] = int(digits[0]) if digits else None
    out["smoke"] = "smoke" in parts
    if out["smoke"]:
        out["epochs"] = 2
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    import lcpr_train as T

    ap = argparse.ArgumentParser(description="sequential candidate queue (one GPU)")
    ap.add_argument("--purpose", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("jobs", nargs="+")
    args = ap.parse_args(argv)

    log_path = C.ANALYSIS / f"queue_{args.purpose}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    for spec in args.jobs:
        job = parse_spec(spec)
        if job["smoke"]:
            out_dir = C.RUNS / "smoke" / f"{job['candidate']}__{job['dataset']}"
        else:
            out_dir = C.run_dir(job["candidate"], job["dataset"])
        done = out_dir / "train.json"
        if done.exists() and not job["smoke"]:
            payload = json.loads(done.read_text())
            if payload.get("status") == "completed":
                C.log(f"[queue:{args.purpose}] skip {spec} (already completed)")
                continue
        t0 = time.perf_counter()
        record: dict[str, Any] = {"purpose": args.purpose, "job": job, "started_utc": C.utcnow(),
                                  "device": str(device), "pid": __import__("os").getpid()}
        try:
            payload = T.train_candidate(candidate=job["candidate"], dataset=job["dataset"],
                                        device=device, out_dir=out_dir,
                                        recipe_name=job["recipe"], epochs_override=job["epochs"],
                                        smoke=bool(job["smoke"]))
            record.update({"status": "completed",
                           "valid": payload["metrics"]["best"]["valid"],
                           "test": payload["metrics"]["best"]["test"],
                           "parameters": payload["model"]["parameters"]})
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()[-4000:]})
            C.log(f"[queue:{args.purpose}] FAILED {spec}: {type(exc).__name__}: {exc}")
        wall = time.perf_counter() - t0
        record.update({"wall_seconds": wall,
                       "gpu_hours_charged": wall / 3600.0 if device.type == "cuda" else 0.0,
                       "finished_utc": C.utcnow()})
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
        if record["status"] == "failed" and "infra" in str(record.get("error", "")).lower():
            raise SystemExit(1)
    C.log(f"[queue:{args.purpose}] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
