"""Write ``manifest.json`` (file list + SHA-256) for the round directory.

    python esr_manifest.py

The manifest covers every commit-worthy file of the round (code, configs, tests,
analysis JSON/CSV, PROTOCOL.md, REPORT.md, figures) plus the head commit and the
hash of each run's ``run.json`` (the run outputs themselves are gitignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import esr_common as C

SKIP_DIRS = {"runs", "cache", "__pycache__", ".pytest_cache"}
SKIP_SUFFIX = {".pt", ".npy", ".npz", ".png", ".pdf"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix in SKIP_SUFFIX:
            continue
        out.append({"path": str(rel), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return out


def runs_digest(root: Path) -> list[dict[str, Any]]:
    out = []
    runs = root / "runs"
    if not runs.exists():
        return out
    for run_json in sorted(runs.glob("*/run.json")):
        payload = json.loads(run_json.read_text())
        if payload.get("smoke"):
            continue
        out.append(
            {
                "run_id": payload.get("run_id"),
                "dataset": payload.get("dataset"),
                "arm": payload.get("arm"),
                "best_epoch": payload.get("best_epoch"),
                "parameters": payload.get("parameters"),
                "gpu_hours": payload.get("gpu_hours"),
                "run_json_sha256": sha256_file(run_json),
                "curve_sha256": sha256_file(run_json.parent / "curve.csv") if (run_json.parent / "curve.csv").exists() else None,
            }
        )
    return out


def git_head(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain"))}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="round manifest")
    ap.add_argument("--root", default=str(C.ROUND))
    args = ap.parse_args(argv)
    root = Path(args.root)
    payload = {
        "round_id": C.ROUND_ID,
        "t0_utc": C.T0_UTC,
        "generated_utc": C.utcnow(),
        "elapsed_hours": C.elapsed_hours(),
        "git": git_head(C.REPO),
        "files": collect(root),
        "runs": runs_digest(root),
    }
    C.write_json(root / "manifest.json", payload)
    C.log(f"manifest: {len(payload['files'])} files, {len(payload['runs'])} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
