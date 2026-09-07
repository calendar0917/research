"""JSON/YAML serialization, atomic writes and content hashing.

All files written by the run control plane go through the atomic path here:
content is first written to a sibling temporary file and then moved into
place with :func:`os.replace`, so readers never observe a half-written file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

_TMP_SUFFIX = ".runtime-tmp"


def jsonable(value: Any) -> Any:
    """Recursively convert a value to plain JSON types.

    Handles ``Path``, ``bytes``, numpy scalars/arrays, mappings and nested
    sequences.  ``bytes`` are hex encoded to keep the output printable.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"cty": "bytes_hex", "data": value.hex()}
    if isinstance(value, bool):  # bool is an int subclass; guard first
        return bool(value)
    # numpy objects handled lazily: runtime is installable without numpy.
    if value.__class__.__module__.split(".")[0] == "numpy":
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return json.dumps(value, allow_nan=True)
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    # Last resort: stringify instead of silently dropping scientific data.
    return str(value)


def dumps_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def dumps_yaml(payload: Any) -> str:
    return yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + _TMP_SUFFIX)
    try:
        temporary.write_text(text, encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    _atomic_write(Path(path), dumps_json(jsonable(payload)))


def write_yaml_atomic(path: Path, payload: Any) -> None:
    _atomic_write(Path(path), dumps_yaml(jsonable(payload)))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(chunk)
    return digest.hexdigest()
