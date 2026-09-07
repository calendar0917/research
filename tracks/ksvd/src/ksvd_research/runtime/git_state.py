"""Git state capture: commit, dirty flag, diff hash, dirty patch, code identity.

Every function is defensive: if the repository or git binary is unavailable
the state degrades to ``{commit: null, dirty: false, diff_hash: "", patch: ""}``
instead of crashing a run.

``code_state_hash`` is the run's code identity.  It must include untracked
*source files* (new code that has never been ``git add``-ed yet still shapes
the science), not just tracked diffs::

    code_state_hash = sha256(canonical_json({
        "commit": ...,
        "tracked_diff_hash": ...,
        "untracked": [[path, sha256], ...],   # sorted, deterministic
    }))

Untracked text files that clearly look like code/config are additionally
snapshotted (``snapshot=true``) inside the run directory under ``git.untracked/``
so a future rebuild still knows what code produced the run.  Paths are
validated against the repository root; over-size or non-code files keep
``sha256``+``size`` but are never snapshotted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .paths import REPO_ROOT, repo_root
from .serialization import dumps_json, jsonable, sha256_file, sha256_text

# Text files that are clearly code / configuration, eligible for snapshots.
SNAPSHOT_EXTENSIONS = (
    ".py",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".sh",
    ".md",
)
MAX_SNAPSHOT_FILE_BYTES = 1 * 1024 * 1024  # per-file snapshot cap
MAX_SNAPSHOT_TOTAL_BYTES = 10 * 1024 * 1024  # total snapshot budget per run
MAX_HASH_FILE_BYTES = 200 * 1024 * 1024  # never hash pathological files


@dataclass(frozen=True)
class UntrackedFile:
    """One untracked file: identity (path + SHA-256) and snapshot policy."""

    path: str
    sha256: str | None
    size: int | None
    snapshot: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GitState:
    commit: str | None
    dirty: bool
    diff_hash: str
    patch: str
    untracked: tuple[UntrackedFile, ...]
    code_state_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd or repo_root()), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def current_commit() -> str | None:
    output = _git("rev-parse", "HEAD")
    if output is None:
        return None
    return output.strip() or None


def porcelain_status() -> str | None:
    """NUL-separated porcelain status (robust against odd path names)."""
    return _git("status", "--porcelain=v1", "-z", "--untracked-files=all")


def is_dirty(status: str | None) -> bool:
    return bool(status and status.strip())


def dirty_patch() -> str:
    """Diff of tracked content (staged + unstaged) against HEAD."""
    part = _git("diff", "HEAD", "--no-ext-diff") or ""
    return part


def untracked_paths(status: str | None) -> tuple[str, ...]:
    if not status:
        return ()
    paths: list[str] = []
    entries = status.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        if len(entry) < 4:
            index += 1
            continue
        marker = entry[:2]
        if marker == "??":
            paths.append(entry[3:])
        elif marker[0] in "RC":
            # rename/copy: a second (destination) path entry follows
            index += 1
        index += 1
    return tuple(paths)


def diff_hash(patch: str) -> str:
    if not patch:
        return ""
    return sha256_text(patch)


def _in_repo_root(path: Path) -> bool:
    root = REPO_ROOT.resolve()
    try:
        return path.resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _is_snapshot_name(path: str) -> bool:
    lowered = path.lower()
    return any(lowered.endswith(suffix) for suffix in SNAPSHOT_EXTENSIONS)


def _untracked_entries(paths: Iterable[str]) -> tuple[UntrackedFile, ...]:
    """Build ordered untracked-file records: path, sha256, size, snapshot flag."""
    entries: list[UntrackedFile] = []
    total_snapshot_bytes = 0
    for raw_path in sorted(paths):
        relative = Path(raw_path)
        source = (REPO_ROOT / relative).resolve()
        safe = _in_repo_root(source)
        size: int | None = None
        digest: str | None = None
        if safe and source.is_file():
            try:
                size = source.stat().st_size
            except OSError:
                size = None
            if size is not None and size <= MAX_HASH_FILE_BYTES:
                try:
                    digest = sha256_file(source)
                except OSError:
                    digest = None
        snapshot = bool(
            safe
            and source.is_file()
            and size is not None
            and size <= MAX_SNAPSHOT_FILE_BYTES
            and _is_snapshot_name(raw_path)
            and total_snapshot_bytes + size <= MAX_SNAPSHOT_TOTAL_BYTES
        )
        if snapshot:
            total_snapshot_bytes += size
        entries.append(
            UntrackedFile(
                path=raw_path,
                sha256=digest,
                size=size,
                snapshot=snapshot,
            )
        )
    return tuple(entries)


def compute_code_state_hash(
    commit: str | None,
    diff_hash_value: str,
    untracked: Iterable[UntrackedFile | Mapping[str, Any] | str],
) -> str:
    payload = {
        "commit": commit or "unknown",
        "tracked_diff_hash": diff_hash_value or "",
        "untracked": _normalized_untracked(untracked),
    }
    return sha256_text(dumps_json(jsonable(payload)))


def _normalized_untracked(
    untracked: Iterable[UntrackedFile | Mapping[str, Any] | str],
) -> list[list[str]]:
    items: list[list[str]] = []
    for entry in untracked:
        if isinstance(entry, UntrackedFile):
            items.append([entry.path, str(entry.sha256 or "")])
        elif isinstance(entry, Mapping):
            items.append([str(entry.get("path") or ""), str(entry.get("sha256") or "")])
        else:
            items.append([str(entry), ""])
    return sorted(items)


def code_state_hash_from_git_dict(git: Mapping[str, Any]) -> str:
    """Recompute the code-state hash from a JSON-ish git dict.

    Handles both the structured ``untracked`` entries (path/sha256) and the
    legacy string-only entries (hash treated as unknown).
    """
    return compute_code_state_hash(
        git.get("commit"),
        git.get("diff_hash") or "",
        git.get("untracked") or [],
    )


def capture_git_state() -> GitState:
    status = porcelain_status()
    patch = dirty_patch()
    untracked = _untracked_entries(untracked_paths(status))
    state_hash = compute_code_state_hash(
        current_commit(),
        diff_hash(patch),
        untracked,
    )
    return GitState(
        commit=current_commit(),
        dirty=is_dirty(status),
        diff_hash=diff_hash(patch),
        patch=patch,
        untracked=untracked,
        code_state_hash=state_hash,
    )


def write_untracked_snapshots(
    run_dir: Path,
    untracked: Iterable[UntrackedFile],
) -> list[Path]:
    """Copy ``snapshot=true`` files into ``<run_dir>/git.untracked/``.

    Only files whose resolved path stays inside the repository root and whose
    target stays inside the snapshot root are copied; everything else is
    recorded by hash only.
    """
    snapshot_root = (Path(run_dir) / "git.untracked").resolve()
    written: list[Path] = []
    for entry in untracked:
        if not entry.snapshot or not entry.sha256:
            continue
        source = (REPO_ROOT / entry.path).resolve()
        if not _in_repo_root(source) or not source.is_file():
            continue
        target = (snapshot_root / entry.path).resolve()
        if not target.is_relative_to(snapshot_root):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_bytes(source.read_bytes())
        except OSError:
            continue
        written.append(target)
    return written
