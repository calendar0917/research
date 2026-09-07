"""Run manifest schema and writing helpers.

The manifest is the machine-readable record of one run.  It is written
synchronously at start (status ``"running"``) and rewritten at the end with
status ``"completed"`` or ``"failed"`` plus metrics and runtime.  Manifest
files live inside the run directory, which is git-ignored; promoted records
under ``records/runs/`` are the git-tracked projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import resolve_path
from .serialization import load_json, write_json_atomic

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"

RUN_FILE_LAYOUT = {
    "manifest": "manifest.json",
    "config": "config.resolved.yaml",
    "protocol": "protocol.snapshot.yaml",
    "metrics": "metrics.json",
    "stdout": "stdout.log",
    "stderr": "stderr.log",
    "patch": "git.diff.patch",
    "untracked": "git.untracked/",
    "artifacts": "artifacts/",
}


@dataclass
class RunSpec:
    """Everything the CLI knows about a run before it executes."""

    run_id: str
    study_id: str
    candidate_id: str
    protocol_id: str
    mode: str
    purpose: str
    runner: str
    config_hash: str
    dataset_fingerprint: dict[str, Any]
    split_fingerprint: str
    seeds: list[int]
    test_access: str
    fingerprint: str
    run_dir: Path
    git: dict[str, Any]
    environment: dict[str, Any]


@dataclass
class RunResult:
    """Returned by a runner function."""

    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class RunContext:
    """What a runner function receives at execution time."""

    run_id: str
    run_dir: Path
    artifact_dir: Path
    mode: str
    protocol: dict[str, Any] | None
    study: dict[str, Any] | None


def build_manifest(
    *,
    spec: RunSpec,
    started_at: datetime,
    status: str,
    metrics: dict[str, Any] | None = None,
    ended_at: datetime | None = None,
    runtime_seconds: float | None = None,
    error: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    signed = {
        "manifest_version": MANIFEST_VERSION,
        "run_id": spec.run_id,
        "study_id": spec.study_id,
        "candidate_id": spec.candidate_id,
        "protocol_id": spec.protocol_id,
        "mode": spec.mode,
        "purpose": spec.purpose,
        "runner": spec.runner,
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": None if ended_at is None else ended_at.isoformat(timespec="seconds"),
        "status": status,
        "config_hash": spec.config_hash,
        "git": spec.git,
        "environment": spec.environment,
        "dataset_fingerprint": spec.dataset_fingerprint,
        "split_fingerprint": spec.split_fingerprint,
        "seeds": spec.seeds,
        "test_access": spec.test_access,
        "fingerprint": spec.fingerprint,
        "metrics": metrics or {},
        "runtime_seconds": runtime_seconds,
        "error": error,
        "files": dict(RUN_FILE_LAYOUT),
    }
    if config is not None:
        signed["config_sample_keys"] = sorted(str(key) for key in config)
    return signed


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    write_json_atomic(resolve_path(run_dir) / MANIFEST_NAME, manifest)
    return manifest


def load_manifest(run_dir: Path | str) -> dict[str, Any]:
    path = Path(run_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no manifest at {path}")
    return load_json(path)
