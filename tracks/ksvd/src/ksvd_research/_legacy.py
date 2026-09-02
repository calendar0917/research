"""Load historical KSVD modules without importing the top-level ``code`` name.

The first migration step keeps the exact historical implementations and
exposes them through package-qualified wrappers.  The loader gives the old
directory an internal package name, so relative imports continue to work even
when the project is run from an installed editable environment where the
repository root is not on ``sys.path``.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType


_PACKAGE_NAME = "ksvd_research._legacy_code"


def _legacy_path() -> Path:
    source_path = Path(__file__).resolve().parents[2] / "code"
    bundled_path = Path(__file__).resolve().parent / "_legacy_code"
    if bundled_path.is_dir():
        return bundled_path
    return source_path


def _load_legacy_package() -> ModuleType:
    loaded = sys.modules.get(_PACKAGE_NAME)
    if loaded is not None:
        return loaded
    root = _legacy_path()
    init_file = root / "__init__.py"
    if not init_file.is_file():
        raise ImportError(f"historical KSVD package is not available at {root}")
    spec = spec_from_file_location(
        _PACKAGE_NAME,
        init_file,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load historical KSVD package from {root}")
    module = module_from_spec(spec)
    sys.modules[_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


def import_legacy(module_name: str) -> ModuleType:
    """Return one historical module under the private migration namespace."""

    # In a wheel the selected compatibility modules are regular package files
    # and can be imported directly (zipimport does not expose them as a local
    # directory).  In an editable checkout the private package is absent, so
    # fall back to loading the unchanged historical directory under the same
    # qualified name.
    try:
        return import_module(f"{_PACKAGE_NAME}.{module_name}")
    except ModuleNotFoundError as exc:
        if exc.name != _PACKAGE_NAME:
            raise
    _load_legacy_package()
    return import_module(f"{_PACKAGE_NAME}.{module_name}")
