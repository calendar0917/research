"""Repository and control-plane path resolution.

The package lives at ``<repo>/tracks/ksvd/src/ksvd_research``.  All paths
exported here are computed from the package location instead of ``cwd`` so
runs remain reproducible regardless of where the CLI is invoked from.

Run directories follow ``runs/YYYY/MM/DD/<run_id>/`` under the track root and
are git-ignored; promoted records live under ``records/`` and are tracked.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve()
# <repo>/tracks/ksvd/src/ksvd_research/runtime/paths.py
REPO_ROOT = _PACKAGE_DIR.parents[5]
TRACK_ROOT = _PACKAGE_DIR.parents[3]  # <repo>/tracks/ksvd


def repo_root() -> Path:
    return REPO_ROOT


def track_root() -> Path:
    return TRACK_ROOT


def runs_root() -> Path:
    return TRACK_ROOT / "runs"


def records_root() -> Path:
    return TRACK_ROOT / "records"


def state_file() -> Path:
    return TRACK_ROOT / "STATE.yaml"


def protocols_root() -> Path:
    return TRACK_ROOT / "protocols"


def studies_root() -> Path:
    return TRACK_ROOT / "studies"


def resolve_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root, like legacy `_resolve`."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def run_directory(run_id: str, created: datetime | None = None) -> Path:
    when = created or datetime.now()
    return runs_root() / f"{when:%Y}" / f"{when:%m}" / f"{when:%d}" / run_id


def date_dir_of(run_dir: Path) -> tuple[str, str, str]:
    current = Path(run_dir)
    date_path = current.parent
    return date_path.name, date_path.parent.name, current.parents[1].name
