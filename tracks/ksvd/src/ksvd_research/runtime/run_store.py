"""The run store: run IDs, directories, duplicate detection, promotion.

Only promoted runs are git-tracked.  Everything in ``runs/`` is ignored, so
scratch runs never create Git noise.
"""

from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from .manifest import MANIFEST_NAME
from .paths import records_root, run_directory, runs_root
from .serialization import (
    dumps_json,
    jsonable,
    load_json,
    sha256_text,
    write_json_atomic,
    write_yaml_atomic,
)


def new_run_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"


def create_run_directory(run_id: str, created: datetime | None = None) -> Path:
    root = run_directory(run_id, created)
    root.mkdir(parents=True, exist_ok=False)
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    return root


def run_fingerprint(
    *,
    runner: str,
    study_id: str,
    protocol_id: str,
    candidate_id: str,
    config: dict[str, Any],
    seeds: list[int],
    git: dict[str, Any],
    dataset_fingerprint: dict[str, Any],
    split_fingerprint: str,
) -> str:
    """Duplicate fingerprint: runner + scientific config + protocol + seed + code state."""
    from .config import scientific_config

    payload = {
        "runner": runner,
        "study_id": study_id,
        "protocol_id": protocol_id,
        "candidate_id": candidate_id,
        "scientific_config": scientific_config(config),
        "seeds": seeds,
        "git_commit": git.get("commit") or "unknown",
        "git_diff_hash": git.get("diff_hash") or "",
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
    }
    return sha256_text(dumps_json(jsonable(payload)))


def locate_manifest(run_id: str) -> Path:
    """Locate a run manifest independent of its date directory."""
    for year in runs_root().glob("*"):
        for month in year.glob("*"):
            for candidate in month.glob("*"):
                manifest = candidate / MANIFEST_NAME
                if manifest.is_file() and candidate.name == run_id:
                    return manifest
    return Path(runs_root()) / "unknown-date" / run_id / MANIFEST_NAME


def load_run(run_id: str) -> dict[str, Any]:
    """Load a run manifest by exact run ID."""
    manifest_path = locate_manifest(run_id)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run {run_id!r} not found (looked at {manifest_path})")
    return load_json(manifest_path)


def list_runs(
    study_id: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for manifest in sorted(runs_root().glob("*/*/*/manifest.json")):
        if manifest.parent.name.startswith("."):
            continue
        try:
            payload = load_json(manifest)
        except OSError:
            continue
        runs.append(payload)
    if study_id is not None:
        runs = [run for run in runs if run.get("study_id") == study_id]
    if status is not None:
        runs = [run for run in runs if run.get("status") == status]
    runs.sort(key=lambda run: run.get("started_at") or "", reverse=True)
    if limit is not None:
        runs = runs[: int(limit)]
    return runs


def find_duplicate_run(runs: list[dict[str, Any]], fingerprint: str) -> dict[str, Any] | None:
    for run in runs:
        if run.get("status") == "completed" and run.get("fingerprint") == fingerprint:
            return run
    return None


def read_metrics_file(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "metrics.json"
    if not path.is_file():
        return {}
    return load_json(path)


def write_metrics_file(run_dir: Path, metrics: dict[str, Any]) -> None:
    write_json_atomic(Path(run_dir) / "metrics.json", metrics)


def write_config_resolved(run_dir: Path, config: dict[str, Any]) -> None:
    write_yaml_atomic(Path(run_dir) / "config.resolved.yaml", config)


def write_patch(run_dir: Path, patch: str) -> None:
    if not patch:
        return
    (Path(run_dir) / "git.diff.patch").write_text(patch, encoding="utf-8")


# ----------------------------------------------------------------------
# Promotion: the git-tracked projection of a finished run.
# ----------------------------------------------------------------------


def record_path_for_run(run_id: str) -> Path:
    return records_root() / "runs" / f"{run_id}.json"


def promote_run(
    manifest: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    run_dir: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an immutable promoted record for a completed run.

    A record is small on purpose: it stores the manifest identity, the
    scientific config, key metrics and provenance pointers, not the bulky
    legacy outputs.  Re-promotion with identical content is a no-op;
    anything else fails loudly (records are immutable).
    """
    from .config import scientific_config

    record = {
        "kind": "run_record",
        "record_id": f"record-{manifest['run_id']}",
        "run_id": manifest["run_id"],
        "study_id": manifest.get("study_id"),
        "protocol_id": manifest.get("protocol_id"),
        "candidate_id": manifest.get("candidate_id"),
        "mode": manifest.get("mode"),
        "purpose": manifest.get("purpose"),
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "status": manifest.get("status"),
        "seeds": manifest.get("seeds") or [],
        "metrics": manifest.get("metrics") or {},
        "scientific_config": scientific_config(config) if config else {},
        "provenance": {
            "runner": manifest.get("runner"),
            "config_hash": manifest.get("config_hash"),
            "git": manifest.get("git"),
            "dataset_fingerprint": manifest.get("dataset_fingerprint"),
            "split_fingerprint": manifest.get("split_fingerprint"),
            "environment": manifest.get("environment"),
            "test_access": manifest.get("test_access"),
            "runtime_seconds": manifest.get("runtime_seconds"),
        },
    }
    if meta:
        record["meta"] = meta
    path = record_path_for_run(manifest["run_id"])
    if path.is_file():
        existing = load_json(path)
        existing.pop("promoted_at", None)
        incoming = dict(record)
        incoming.pop("promoted_at", None)
        if existing != incoming:
            raise RuntimeError(
                f"record {path} already exists with different content; records are immutable"
            )
        return record
    write_json_atomic(path, record)
    return record


def promoted_record(run_id: str) -> dict[str, Any]:
    path = record_path_for_run(run_id)
    if not path.is_file():
        raise FileNotFoundError(f"no promoted record for run {run_id!r}")
    return load_json(path)
