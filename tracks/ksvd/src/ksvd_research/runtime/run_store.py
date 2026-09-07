"""The run store: local runs, promoted records, duplicate detection, promotion.

Two explicit run sources exist:

* **local run** — a resolved execution snapshot under ``runs/`` (git-ignored).
  It is the authoritative execution view: resolved config, stdout/stderr,
  artifacts, git diff patch, untracked-code snapshots.
* **promoted record** — a small immutable JSON under ``records/runs/``
  (git-tracked).  It is the durable scientific fact projection that survives
  a fresh clone or a different machine.

``list_runs`` / ``load_run`` default to ``source="all"`` so the CLI sees both,
deduplicated by run id (local view preferred, flagged ``promoted: true``).
Schema differences between the two live here in ``run_view_from_record``; the
CLI only ever handles manifest-like dicts.
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

RUN_SOURCES = ("all", "local", "promoted")


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
    protocol_hash: str | None = None,
) -> str:
    """Duplicate fingerprint: runner + scientific config + protocol + seed + code state.

    Code identity prefers ``code_state_hash`` (commit + tracked diff hash +
    ordered untracked file SHA-256s).  ``git_commit`` / ``git_diff_hash`` are
    kept for readability only.
    """
    from .config import scientific_config
    from .git_state import code_state_hash_from_git_dict

    payload = {
        "runner": runner,
        "study_id": study_id,
        "protocol_id": protocol_id,
        "candidate_id": candidate_id,
        "scientific_config": scientific_config(config),
        "seeds": seeds,
        "code_state_hash": git.get("code_state_hash")
        or code_state_hash_from_git_dict(git),
        "protocol_hash": protocol_hash or "legacy-unknown",
        "git_commit": git.get("commit") or "unknown",
        "git_diff_hash": git.get("diff_hash") or "",
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
    }
    return sha256_text(dumps_json(jsonable(payload)))


# ----------------------------------------------------------------------
# local runs
# ----------------------------------------------------------------------


def _local_manifests() -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not runs_root().is_dir():
        return manifests
    for manifest in sorted(runs_root().glob("*/*/*/*/manifest.json")):
        if manifest.parent.name.startswith("."):
            continue
        try:
            payload = load_json(manifest)
        except OSError:
            continue
        if not isinstance(payload, dict):
            continue
        manifests.append(payload)
    return manifests


def list_local_runs(
    study_id: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    runs = _local_manifests()
    if study_id is not None:
        runs = [run for run in runs if run.get("study_id") == study_id]
    if status is not None:
        runs = [run for run in runs if run.get("status") == status]
    runs.sort(key=lambda run: run.get("started_at") or "", reverse=True)
    if limit is not None:
        runs = runs[: int(limit)]
    return runs


def locate_manifest(run_id: str) -> Path:
    """Locate a local run manifest independent of its date directory."""
    for year in runs_root().glob("*"):
        for month in year.glob("*"):
            for day in month.glob("*"):
                for candidate in day.glob("*"):
                    manifest = candidate / MANIFEST_NAME
                    if manifest.is_file() and candidate.name == run_id:
                        return manifest
    return Path(runs_root()) / "unknown-date" / run_id / MANIFEST_NAME


def load_local_run(run_id: str) -> dict[str, Any]:
    """Load the local execution manifest by exact run ID (no fallback)."""
    manifest_path = locate_manifest(run_id)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"local run {run_id!r} not found (looked at {manifest_path})")
    return load_json(manifest_path)


# ----------------------------------------------------------------------
# promoted records
# ----------------------------------------------------------------------


def run_view_from_record(record: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    """Project a git-tracked promoted record into a manifest-like RunView.

    Field differences (``provenance`` nesting, missing keys) are resolved
    here so the CLI never has to branch on record vs manifest schemas.
    Every missing optional field degrades to ``None``/``[]`` — never to a
    guessed value.
    """
    provenance = record.get("provenance") or {}
    if not isinstance(provenance, dict):
        provenance = {}
    git = provenance.get("git") or {}
    if not isinstance(git, dict):
        git = {}
    return {
        "run_id": record.get("run_id"),
        "record_id": record.get("record_id"),
        "study_id": record.get("study_id"),
        "protocol_id": record.get("protocol_id"),
        "protocol_hash": record.get("protocol_hash"),
        "candidate_id": record.get("candidate_id"),
        "mode": record.get("mode"),
        "purpose": record.get("purpose"),
        "status": record.get("status"),
        "seeds": record.get("seeds") or [],
        "metrics": record.get("metrics") or {},
        "fingerprint": record.get("fingerprint"),
        "runner": provenance.get("runner"),
        "git": git,
        "environment": provenance.get("environment"),
        "dataset_fingerprint": provenance.get("dataset_fingerprint"),
        "split_fingerprint": provenance.get("split_fingerprint")
        or record.get("split_fingerprint"),
        "test_access": provenance.get("test_access"),
        "config_hash": provenance.get("config_hash"),
        "runtime_seconds": provenance.get("runtime_seconds"),
        "started_at": record.get("started_at"),
        "ended_at": record.get("ended_at"),
        "source": "promoted",
        "promoted": True,
        "record_path": str(path) if path is not None else None,
    }


def _promoted_record_paths() -> list[Path]:
    root = records_root() / "runs"
    if not root.is_dir():
        return []
    return sorted(root.glob("*.json"))


def list_promoted_records(
    study_id: str | None = None,
    status: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _promoted_record_paths():
        try:
            record = load_json(path)
        except OSError:
            continue
        if not isinstance(record, dict) or not record.get("run_id"):
            continue
        records.append(run_view_from_record(record, path=path))
    if study_id is not None:
        records = [record for record in records if record.get("study_id") == study_id]
    if status is not None:
        records = [record for record in records if record.get("status") == status]
    records.sort(key=lambda record: record.get("started_at") or "", reverse=True)
    if limit is not None:
        records = records[: int(limit)]
    return records


def load_promoted_record(run_id: str) -> dict[str, Any]:
    path = record_path_for_run(run_id)
    if not path.is_file():
        raise FileNotFoundError(f"no promoted record for run {run_id!r}")
    return load_json(path)


def promoted_record(run_id: str) -> dict[str, Any]:
    return load_promoted_record(run_id)


# ----------------------------------------------------------------------
# unified view
# ----------------------------------------------------------------------


def list_runs(
    study_id: str | None = None,
    status: str | None = None,
    limit: int | None = None,
    source: str = "all",
) -> list[dict[str, Any]]:
    """List runs from ``all`` / ``local`` / ``promoted`` sources.

    With ``source="all"`` the local manifest is the detailed execution view;
    when a promoted record exists for the same run id the row is flagged
    ``promoted: true`` instead of being duplicated.
    """
    if source not in RUN_SOURCES:
        raise ValueError(f"unknown run source {source!r}; expected one of {RUN_SOURCES}")
    local_views = list_local_runs() if source in ("all", "local") else []
    promoted_views = list_promoted_records()

    in_local = {view["run_id"]: view for view in local_views}

    merged: dict[str, dict[str, Any]] = {}
    for view in local_views:
        row = dict(view)
        row["source"] = "local"
        row["promoted"] = False
        record_view = promoted_by_run_id(view["run_id"], promoted_views)
        if record_view is not None:
            row["promoted"] = True
            row["record_path"] = record_view.get("record_path")
        merged[view["run_id"]] = row
    if source in ("all", "promoted"):
        for view in promoted_views:
            if view["run_id"] not in in_local:
                merged[view["run_id"]] = view

    runs = list(merged.values())
    if study_id is not None:
        runs = [run for run in runs if run.get("study_id") == study_id]
    if status is not None:
        runs = [run for run in runs if run.get("status") == status]
    runs.sort(key=lambda run: run.get("started_at") or "", reverse=True)
    if limit is not None:
        runs = runs[: int(limit)]
    return runs


def promoted_by_run_id(
    run_id: str, views: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for view in views:
        if view.get("run_id") == run_id:
            return view
    return None


def load_run(run_id: str, source: str = "all") -> dict[str, Any]:
    """Load a run view: local manifest first, promoted record as fallback."""
    if source not in RUN_SOURCES:
        raise ValueError(f"unknown run source {source!r}; expected one of {RUN_SOURCES}")
    if source == "promoted":
        return run_view_from_record(load_promoted_record(run_id))
    try:
        manifest = load_local_run(run_id)
    except FileNotFoundError:
        if source == "local":
            raise
        record = load_promoted_record(run_id)
        view = run_view_from_record(record)
        view["source"] = "promoted"
        return view
    view = dict(manifest)
    view["source"] = "local"
    if source == "all":
        record_path = record_path_for_run(run_id)
        if record_path.is_file():
            view["promoted"] = True
            view["record_path"] = str(record_path)
        else:
            view["promoted"] = False
    return view


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
# Promotion: the git-tracked projection of a finished local run.
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
    """Write an immutable promoted record for a completed local run.

    A record is small on purpose: it stores the manifest identity, the
    scientific config, key metrics, the code-state identity and provenance
    pointers, not the bulky legacy outputs.  Re-promotion with identical
    content is a no-op; anything else fails loudly (records are immutable).
    New records additionally carry ``fingerprint``, ``started_at``,
    ``ended_at`` and ``protocol_hash`` for durable identity; old records that
    lack them are left untouched (backward compatible).
    """
    from .config import scientific_config

    record = {
        "kind": "run_record",
        "record_id": f"record-{manifest['run_id']}",
        "run_id": manifest["run_id"],
        "study_id": manifest.get("study_id"),
        "protocol_id": manifest.get("protocol_id"),
        "protocol_hash": manifest.get("protocol_hash"),
        "candidate_id": manifest.get("candidate_id"),
        "mode": manifest.get("mode"),
        "purpose": manifest.get("purpose"),
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "status": manifest.get("status"),
        "seeds": manifest.get("seeds") or [],
        "fingerprint": manifest.get("fingerprint"),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "metrics": manifest.get("metrics") or {},
        "scientific_config": scientific_config(config) if config else {},
        "provenance": {
            "runner": manifest.get("runner"),
            "config_hash": manifest.get("config_hash"),
            "git": manifest.get("git"),
            "code_state_hash": (manifest.get("git") or {}).get("code_state_hash"),
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
