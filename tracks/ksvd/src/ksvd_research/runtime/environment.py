"""Environment capture for run manifests.

The set of packages is fixed and best-effort: missing packages are recorded
as ``null`` rather than crashing a run.  Only versions are recorded; paths
that vary between machines are captured as machine facts where relevant.
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "pyyaml",
    "torch",
    "torch-geometric",
    "ogb",
    "networkx",
    "pynauty",
    "optuna",
    "pytest",
)


def dependency_versions(names=TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capture_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": str(Path.cwd()),
        "packages": {
            name: version
            for name, version in dependency_versions().items()
            if version is not None
        },
    }
