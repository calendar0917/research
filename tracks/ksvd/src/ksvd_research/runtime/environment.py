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

# Hard dependencies of the active zinc runner chain (+ the control plane's
# own yaml).  Missing here is FAIL.
REQUIRED_PACKAGES = (
    "pyyaml",
    "numpy",
    "scikit-learn",
    "torch",
    "torch-geometric",
    "pynauty",
)

# Standalone / other-track / optional tooling.  Missing here is WARN only.
OPTIONAL_PACKAGES = ("scipy", "pandas", "networkx", "ogb", "optuna", "pytest")


def dependency_versions(names=TRACKED_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def capture_environment() -> dict[str, Any]:
    # Keep every TRACKED_PACKAGES key, including nulls: the manifest must
    # express "package absent" explicitly, not "key missing".
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": str(Path.cwd()),
        "packages": dependency_versions(),
    }
