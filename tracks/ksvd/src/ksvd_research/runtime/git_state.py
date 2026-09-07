"""Git state capture: commit, dirty flag, diff hash and dirty patch.

Every function is defensive: if the repository or git binary is unavailable
the state degrades to ``{commit: null, dirty: false, diff_hash: "", patch: ""}``
instead of crashing a run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import subprocess
from pathlib import Path

from .paths import repo_root


@dataclass(frozen=True)
class GitState:
    commit: str | None
    dirty: bool
    diff_hash: str
    patch: str
    untracked: tuple[str, ...]

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
    return _git("status", "--porcelain", "--untracked-files=all")


def is_dirty(status: str | None) -> bool:
    return bool(status and status.strip())


def dirty_patch() -> str:
    """Diff of tracked content (staged + unstaged) against HEAD."""
    part = _git("diff", "HEAD", "--no-ext-diff") or ""
    return part


def untracked_files(status: str | None) -> tuple[str, ...]:
    if not status:
        return ()
    return tuple(
        line[3:]
        for line in status.splitlines()
        if line.startswith("??")
    )


def diff_hash(patch: str) -> str:
    from .serialization import sha256_text

    if not patch:
        return ""
    return sha256_text(patch)


def capture_git_state() -> GitState:
    status = porcelain_status()
    patch = dirty_patch()
    return GitState(
        commit=current_commit(),
        dirty=is_dirty(status),
        diff_hash=diff_hash(patch),
        patch=patch,
        untracked=untracked_files(status),
    )
