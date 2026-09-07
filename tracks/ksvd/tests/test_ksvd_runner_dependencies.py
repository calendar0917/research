"""Dependency boundary test.

New code must keep the direction CLI -> runner -> feature/model/data
components.  Runners may not import other runners (nor the registry package)
and may not import the CLI.  Legacy experiment modules
(``tracks.ksvd.experiments...``) are still allowed in this round because the
runner is a strangler wrapper.
"""

from __future__ import annotations

import ast
from pathlib import Path

RUNNERS_DIR = Path(__file__).resolve().parents[1] / "src" / "ksvd_research" / "runners"
PRIVATE_PREFIXES = ("ksvd_research._legacy",)
CONCRETE_RUNNERS = ("zinc_patch_path_pooling.py",)


def _concrete_runner_paths() -> list[Path]:
    assert RUNNERS_DIR.is_dir(), f"runners directory missing: {RUNNERS_DIR}"
    paths = sorted(
        path
        for path in RUNNERS_DIR.glob("*.py")
        if path.name != "__init__.py"
    )
    assert paths, "no concrete runner modules found; scan must not be empty"
    names = {path.name for path in paths}
    for expected in CONCRETE_RUNNERS:
        assert expected in names, f"expected concrete runner {expected} not scanned"
    return paths


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    collected: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            collected.append(node.module)
        elif isinstance(node, ast.Import):
            collected.extend(alias.name for alias in node.names)
    return collected


def test_runner_directory_exists():
    assert RUNNERS_DIR.is_dir(), f"runners directory must exist at {RUNNERS_DIR}"


def test_runner_scan_is_nonempty():
    paths = _concrete_runner_paths()
    assert any(path.name == "zinc_patch_path_pooling.py" for path in paths)


def test_runner_does_not_import_runner_package():
    for path in _concrete_runner_paths():
        for module in _imports_of(path):
            assert module != "ksvd_research.runners" and not module.startswith(
                "ksvd_research.runners."
            ), f"{path.name} imports the runner registry/package: {module}"


def test_runner_does_not_import_cli_or_cross_areas():
    for path in _concrete_runner_paths():
        for module in _imports_of(path):
            assert not module.startswith("ksvd_research.cli"), (
                f"{path.name} imports the CLI; direction must be CLI -> runner"
            )
            assert not module.startswith(PRIVATE_PREFIXES), (
                f"{path.name} imports legacy private code: {module}"
            )
