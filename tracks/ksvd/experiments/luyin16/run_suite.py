"""Resumable, sequential experiment suite for the luyin16 track.

The suite deliberately runs one task per subprocess.  A long K-SVD/XGBoost
run can therefore be interrupted safely: completed tasks are skipped on the
next invocation, while a failed task is recorded and later tasks continue.
Task commands are represented as Python module invocations in YAML rather than
shell strings, so paths and arguments remain auditable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/suite_long.yaml"
DEFAULT_RESULT_ROOT = REPO_ROOT / "tracks/ksvd/results/luyin16"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"suite configuration must be a mapping: {path}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("suite configuration must contain a non-empty tasks list")
    return payload


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class Task:
    task_id: str
    module: str
    args: tuple[str, ...]
    depends_on: tuple[str, ...]
    env: dict[str, str]
    timeout_sec: int | None
    allow_failure: bool
    description: str

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "Task":
        task_id = str(item.get("id", "")).strip()
        module = str(item.get("module", "")).strip()
        if not task_id or not module:
            raise ValueError("each suite task requires non-empty id and module")
        args = item.get("args", [])
        depends = item.get("depends_on", [])
        if not isinstance(args, list) or not all(isinstance(value, (str, int, float)) for value in args):
            raise ValueError(f"task {task_id}: args must be a list of scalar values")
        if not isinstance(depends, list) or not all(isinstance(value, str) for value in depends):
            raise ValueError(f"task {task_id}: depends_on must be a list of strings")
        env = item.get("env", {})
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError(f"task {task_id}: env must be a string mapping")
        timeout = item.get("timeout_sec")
        if timeout is not None and int(timeout) <= 0:
            raise ValueError(f"task {task_id}: timeout_sec must be positive")
        return cls(
            task_id=task_id,
            module=module,
            args=tuple(str(value) for value in args),
            depends_on=tuple(depends),
            env=dict(env),
            timeout_sec=None if timeout is None else int(timeout),
            allow_failure=bool(item.get("allow_failure", False)),
            description=str(item.get("description", "")),
        )

    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]

    def fingerprint(self) -> str:
        return _sha256_payload(
            {
                "id": self.task_id,
                "module": self.module,
                "args": self.args,
                "depends_on": self.depends_on,
                "env": self.env,
                "timeout_sec": self.timeout_sec,
            }
        )


def _validate_tasks(tasks: Sequence[Task]) -> None:
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("suite task ids must be unique")
    known = set(ids)
    for task in tasks:
        missing = sorted(set(task.depends_on) - known)
        if missing:
            raise ValueError(f"task {task.task_id} depends on unknown task(s): {missing}")
    # A small DFS catches accidental dependency cycles before any process runs.
    visiting: set[str] = set()
    visited: set[str] = set()
    by_id = {task.task_id: task for task in tasks}

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f"dependency cycle includes {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in by_id[task_id].depends_on:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in ids:
        visit(task_id)


def _task_result_path(suite_dir: Path, task_id: str) -> Path:
    return suite_dir / "tasks" / task_id / "status.json"


def _load_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _status_is_reusable(status: Mapping[str, Any] | None, task: Task) -> bool:
    return bool(status and status.get("status") == "success" and status.get("fingerprint") == task.fingerprint())


def _print_task(task: Task, index: int, total: int) -> None:
    print(f"\n[{index}/{total}] {task.task_id}: {task.description or task.module}", flush=True)
    print("  $ " + " ".join(task.command()), flush=True)


def run_suite(
    config_path: Path,
    suite_dir: Path,
    *,
    resume: bool = True,
    dry_run: bool = False,
    only: set[str] | None = None,
    stop_on_failure: bool = False,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    tasks = [Task.from_mapping(item) for item in config["tasks"]]
    _validate_tasks(tasks)
    if only:
        unknown = sorted(only - {task.task_id for task in tasks})
        if unknown:
            raise ValueError(f"--only contains unknown task(s): {unknown}")
    suite_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = suite_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "suite_id": str(config.get("suite_id", config_path.stem)),
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "repo_root": str(REPO_ROOT),
        "git_head": _git_head(),
        "python": sys.version,
        "platform": platform.platform(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "tasks": {},
    }
    _write_json(manifest_path, manifest)

    task_statuses: dict[str, dict[str, Any]] = {}
    executed = 0
    for index, task in enumerate(tasks, start=1):
        if max_tasks is not None and executed >= max_tasks:
            break
        if only and task.task_id not in only:
            continue
        _print_task(task, index, len(tasks))
        status_path = _task_result_path(suite_dir, task.task_id)
        old_status = _load_status(status_path)
        if resume and _status_is_reusable(old_status, task):
            print("  skip: existing successful task with matching fingerprint", flush=True)
            task_statuses[task.task_id] = dict(old_status)
            manifest["tasks"][task.task_id] = task_statuses[task.task_id]
            _write_json(manifest_path, manifest)
            continue

        dependency_blockers = []
        for dependency in task.depends_on:
            dependency_status = task_statuses.get(dependency) or _load_status(_task_result_path(suite_dir, dependency))
            # ``blocked`` is terminal but unavailable: descendants must not
            # start (notably ZINC tasks when its dataset download is absent).
            if not dependency_status or dependency_status.get("status") not in {"success", "dry_run"}:
                dependency_blockers.append(dependency)
        if dependency_blockers:
            task_statuses[task.task_id] = {
                "task_id": task.task_id,
                "fingerprint": task.fingerprint(),
                "status": "blocked",
                "reason": "dependency_failed_or_unavailable",
                "blocked_by": dependency_blockers,
            }
            _write_json(status_path, task_statuses[task.task_id])
            manifest["tasks"][task.task_id] = task_statuses[task.task_id]
            _write_json(manifest_path, manifest)
            print(f"  blocked by: {', '.join(dependency_blockers)}", flush=True)
            continue

        task_dir = status_path.parent
        task_dir.mkdir(parents=True, exist_ok=True)
        if dry_run:
            task_statuses[task.task_id] = {
                "task_id": task.task_id,
                "fingerprint": task.fingerprint(),
                "status": "dry_run",
                "command": task.command(),
            }
            _write_json(status_path, task_statuses[task.task_id])
            manifest["tasks"][task.task_id] = task_statuses[task.task_id]
            _write_json(manifest_path, manifest)
            continue

        started = time.time()
        stdout_path = task_dir / "stdout.log"
        stderr_path = task_dir / "stderr.log"
        env = os.environ.copy()
        env.update(task.env)
        env.setdefault("PYTHONUNBUFFERED", "1")
        payload: dict[str, Any] = {
            "task_id": task.task_id,
            "fingerprint": task.fingerprint(),
            "status": "running",
            "command": task.command(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        _write_json(status_path, payload)
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    task.command(),
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=task.timeout_sec,
                    check=False,
                )
            return_code = int(completed.returncode)
            payload["return_code"] = return_code
            payload["status"] = "success" if return_code == 0 else ("blocked" if task.allow_failure else "failed")
        except subprocess.TimeoutExpired:
            payload["status"] = "timeout"
            payload["return_code"] = None
        except OSError as exc:
            payload["status"] = "failed"
            payload["return_code"] = None
            payload["error"] = repr(exc)
        payload["duration_sec"] = round(time.time() - started, 3)
        payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(status_path, payload)
        task_statuses[task.task_id] = payload
        manifest["tasks"][task.task_id] = payload
        _write_json(manifest_path, manifest)
        executed += 1
        print(f"  {payload['status']} ({payload.get('duration_sec', 0):.1f}s)", flush=True)
        if payload["status"] in {"failed", "timeout"} and stop_on_failure:
            break

    statuses = [task_statuses.get(task.task_id, {}).get("status", "pending") for task in tasks]
    if dry_run:
        manifest["status"] = "dry_run"
    elif any(value in {"failed", "timeout"} for value in statuses):
        manifest["status"] = "failed"
    elif any(value == "pending" for value in statuses):
        manifest["status"] = "partial"
    elif all(value in {"success", "blocked", "dry_run"} for value in statuses):
        manifest["status"] = "complete"
    else:
        manifest["status"] = "partial"
    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["summary"] = {status: statuses.count(status) for status in sorted(set(statuses))}
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="rerun successful tasks")
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    config = _read_yaml(config_path)
    run_name = str(config.get("run_name", time.strftime("suite_%Y%m%d_%H%M%S")))
    suite_dir = _resolve(args.suite_dir) if args.suite_dir else DEFAULT_RESULT_ROOT / run_name
    only = {value.strip() for value in args.only.split(",") if value.strip()} if args.only else None
    manifest = run_suite(
        config_path,
        suite_dir,
        resume=not args.no_resume,
        dry_run=args.dry_run,
        only=only,
        stop_on_failure=args.stop_on_failure,
        max_tasks=args.max_tasks,
    )
    print(json.dumps({"status": manifest["status"], "suite_dir": str(suite_dir), "summary": manifest.get("summary", {})}, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] in {"complete", "dry_run", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
