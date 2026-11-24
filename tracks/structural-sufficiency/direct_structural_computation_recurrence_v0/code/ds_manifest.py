"""DSCR v0 result manifest: paths, SHA256, code commit, GPU-hour ledger.

Run this **after** every training/analysis step of the round; it only reads files
and writes ``results/manifest.json`` plus ``results/ledger.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import ds_common as C  # noqa: E402

ARTIFACT_SUFFIXES = (".json", ".csv", ".npy", ".md", ".txt", ".log")
#: small per-unit artefacts copied into ``results/`` (checkpoints and prediction
#: matrices stay on the host; they are recorded by path + SHA256 only)
STAGE_FILES = (
    "run.json", "run.failed.json", "train_curve.csv", "per_task_ap.csv",
    "best_checkpoint_eval.json", "best_epoch_reports.json", "structural_usage.json",
    "training_order.json",
)
SKIP_DIRS = {"__pycache__", "_shared"}


def sha256(path: Path, *, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        remaining = limit
        while True:
            chunk = fh.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return h.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=C.REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return ""


def walk(base: Path) -> Iterable[Path]:
    if not base.is_dir():
        return []
    out = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            p = Path(root) / name
            if p.suffix in ARTIFACT_SUFFIXES and not name.endswith(".tmp.npy"):
                out.append(p)
    return out


def collect(base: Path) -> list[dict[str, Any]]:
    rows = []
    for p in walk(base):
        rel = p.relative_to(C.REPO)
        rows.append(
            {
                "path": str(rel),
                "bytes": int(p.stat().st_size),
                "sha256": sha256(p),
            }
        )
    return rows


def checkpoints(base: Path) -> list[dict[str, Any]]:
    rows = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            if not name.endswith(".pt"):
                continue
            p = Path(root) / name
            rows.append(
                {
                    "path": str(p.relative_to(C.REPO)),
                    "bytes": int(p.stat().st_size),
                    "sha256": sha256(p),
                    "location": "host-only (never committed)",
                }
            )
    return rows


def stage() -> list[str]:
    """Copy the small per-unit artefacts into ``results/`` (no checkpoints, no logits)."""
    copied: list[str] = []
    for d in sorted(C.RUNS.glob("*_seed*")):
        if not d.is_dir():
            continue
        out = C.RESULTS / "runs" / d.name
        for name in STAGE_FILES:
            src = d / name
            if not src.is_file():
                continue
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / name)
            copied.append(str((out / name).relative_to(C.REPO)))
    hist = C.RUNS / "HIST"
    if hist.is_dir():
        for d in sorted(hist.glob("*_seed*")):
            out = C.RESULTS / "historical_reevaluation" / d.name
            for name in ("best_checkpoint_eval.json",):
                src = d / name
                if src.is_file():
                    out.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, out / name)
                    copied.append(str((out / name).relative_to(C.REPO)))
    return copied


def ledger(units: Sequence[tuple[str, str, int]]) -> list[dict[str, Any]]:
    rows = []
    for label, arm, seed in units:
        d = C.run_dir(arm, seed)
        rj = d / "run.json"
        source = "this round"
        if not rj.is_file() and seed == 0 and arm in C.HISTORICAL:
            d = Path(C.HISTORICAL[arm]["run_dir"])
            rj = d / "run.json"
            source = f"historical {C.HISTORICAL[arm]['protocol']} (reused, not re-run)"
        if not rj.is_file():
            rows.append({"unit": label, "arm": arm, "seed": seed, "status": "missing"})
            continue
        run = C.read_json(rj)
        run_origin = source
        gpus = int(run.get("n_gpus") or 1)
        hours = float(run.get("gpu_hours") or 0.0)
        rows.append(
            {
                "unit": label,
                "arm": arm,
                "seed": seed,
                "status": run.get("status"),
                "horizon": run.get("horizon"),
                "started_utc": run.get("started_utc"),
                "finished_utc": run.get("finished_utc"),
                "wall_hours": round(float(run.get("wall_seconds") or 0.0) / 3600.0, 4),
                "n_gpus": gpus,
                "gpu_hours": round(hours, 4),
                "prior_attempt_gpu_seconds": run.get("gpu_seconds_prior_attempts"),
                "first_started_utc": (run.get("resume_repair") or {}).get("first_started_utc"),
                "best_epoch": run.get("best_epoch"),
                "best_valid_ap": run.get("best_valid_ap"),
                "stopped_reason": run.get("stopped_reason"),
                "origin": run_origin,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.RESULTS / "manifest.json"))
    parser.add_argument("--stage", action="store_true", help="copy the small per-unit artefacts into results/")
    args = parser.parse_args(argv)

    units = [
        ("W2_seed0", "W2", 0),
        ("SELF_seed0", "SELF", 0),
        ("FIXED1_seed0", "FIXED1", 0),
        ("REAL2_seed0", "REAL2", 0),
        ("BAG2_seed0", "BAG2", 0),
        ("DYNAMIC_seed0", "DYNAMIC", 0),
        ("W2_seed1", "W2", 1),
        ("DYNAMIC_seed1", "DYNAMIC", 1),
    ]
    # the seed-1 candidate/control arms are whatever the frozen selection chose
    sel: dict[str, Any] = {}
    if (C.ANALYSIS / "summary.json").is_file():
        sel = C.read_json(C.ANALYSIS / "summary.json").get("candidate_selection") or {}
    for role in ("C*", "K*"):
        arm = sel.get(role)
        if arm:
            units.append((f"{role.replace('*', 'star')}={arm}_seed1", str(arm), 1))
    seen: set[tuple[str, int]] = set()
    uniq: list[tuple[str, str, int]] = []
    for label, arm, seed in units:
        if (arm, seed) in seen:
            continue
        seen.add((arm, seed))
        uniq.append((label, arm, seed))
    units = uniq
    staged = stage() if args.stage else []
    ledger_rows = ledger(units)
    gpu_hours = sum(float(r.get("gpu_hours") or 0.0) for r in ledger_rows if r.get("unit") != "CSTAR_seed1")
    new_units = [r for r in ledger_rows if r.get("status") not in (None, "missing")
                 and not str(r.get("unit", "")).startswith(("W2_seed0", "DYNAMIC_seed0"))]
    new_gpu_hours = sum(float(r.get("gpu_hours") or 0.0) for r in new_units)

    hist = {}
    for arm, info in C.HISTORICAL.items():
        d = Path(info["run_dir"])
        hist[arm] = {
            "run_dir": str(d),
            "run_json_sha256": sha256(d / "run.json") if (d / "run.json").is_file() else None,
            "best_checkpoint_sha256": sha256(d / "checkpoint_best.pt") if (d / "checkpoint_best.pt").is_file() else None,
            "best_epoch": info["best_epoch"],
            "best_valid_ap": info["best_valid_ap"],
            "test_ap_best": info["test_ap_best"],
            "parameters": info["parameters"],
            "reuse": "scored every epoch, never retrained, never used as initialisation",
        }

    payload = {
        "schema": "dscr-results-manifest-v1",
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "code": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_tracked_files": [line for line in git("status", "--porcelain").splitlines()
                                    if line and not line.startswith("??")],
        },
        "environment": {
            "driver": os.environ.get("PI_RR_DRIVER", ""),
            "torch": _torch_version(),
            "node": os.environ.get("PI_RR_NODE", ""),
            "pools": os.environ.get("PI_RR_POOL", ""),
        },
        "horizon": C.read_json(C.CONFIGS / "horizon.json") if (C.CONFIGS / "horizon.json").is_file() else None,
        "artifacts": collect(C.ROUND) + collect(C.RESULTS),
        "checkpoints": checkpoints(C.RUNS),
        "historical_references": hist,
        "ledger": ledger_rows,
        "gpu_hours": {
            "this_round_new_units": round(new_gpu_hours, 4),
            "with_historical_reuse": round(gpu_hours, 4),
            "target": 14.0,
            "hard_cap": 16.0,
            "note": "historical W2/DYNAMIC seed-0 GPU hours are not spent by this round",
        },
        "staged_files": staged,
        "test_role": C.TEST_ROLE,
        "notes": [
            "checkpoints stay on the host (4.5 MB each) and are listed with SHA256 only",
            "artefacts under analysis/ are the frozen outputs of this round",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out, payload)
    with open(C.RESULTS / "ledger.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ledger_rows[0]))
        w.writeheader()
        w.writerows(ledger_rows)
    C.log(f"[manifest] {len(payload['artifacts'])} artefacts, {len(payload['checkpoints'])} checkpoints, "
          f"new gpu_hours {new_gpu_hours:.2f}")
    return 0


def _torch_version() -> str:
    try:
        import torch

        return f"{torch.__version__} (cuda {torch.version.cuda})"
    except Exception:  # pragma: no cover
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
