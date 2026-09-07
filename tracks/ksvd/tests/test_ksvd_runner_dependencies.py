"""Dependency boundary test.

New code must keep the direction CLI -> runner -> feature/model/data
components.  Runners may not import other runners.  Legacy experiment modules
are out of scope for the first round (102+ historical dependencies remain).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RUNNERS_DIR = Path(__file__).resolve().parents[2] / "src" / "ksvd_research" / "runners"
PRIVATE_PREFIXES = ("ksvd_research._legacy",)


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    collected: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            collected.append(node.module)
        elif isinstance(node, ast.Import):
            collected.extend(alias.name for alias in node.names)
    return collected


def test_runner_does_not_import_another_runner():
    for path in sorted(RUNNERS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for module in _imports_of(path):
            if module == "ksvd_research.runners" or module.startswith("ksvd_research.runners."):
                pytest.fail(f"{path.name} imports another runner: {module}")


def test_runner_does_not_import_cli_or_cross_areas():
    for path in sorted(RUNNERS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for module in _imports_of(path):
            assert not module.startswith("ksvd_research.cli"), (
                f"{path.name} imports the CLI; direction must be CLI -> runner"
            )
            assert not module.startswith(PRIVATE_PREFIXES), (
                f"{path.name} imports legacy private code: {module}"
            )
