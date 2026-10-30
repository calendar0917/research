"""Process-level audit hook that denies forbidden label sources.

The round enforces label discipline in three independent ways:

1. structural (separate modules / entry points, checked by static AST tests),
2. data-flow (the training batch contains no label field at all),
3. runtime (this audit hook, which raises ``PermissionError`` if a forbidden
   path is opened while the guard is armed).

The frozen label artifact ``labels_fit_monitor.json`` contains both Fit and
Monitor values.  It is *gated*: the only way to open it is a one-shot grant
issued by :mod:`fit_labels` (Fit keys) or :mod:`monitor_labels` (Monitor keys
after the prediction freeze).  Every other label-bearing source is hard-forbidden.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Iterable

import screen_common as C

_LOCK = threading.Lock()
_STATE: dict = {
    "installed": False,
    "forbidden": set(),
    "gated": set(),
    "grant": None,
    "denied": [],
    "granted": [],
}


def _norm(path) -> str | None:
    try:
        if isinstance(path, int):
            return None
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, (str, os.PathLike)):
            return None
        return os.path.realpath(os.fspath(path))
    except Exception:
        return None


def _hook(event: str, args: tuple) -> None:
    if event != "open" or not args:
        return
    resolved = _norm(args[0])
    if resolved is None:
        return
    with _LOCK:
        grant = _STATE["grant"]
        if grant is not None and resolved == grant:
            _STATE["grant"] = None
            _STATE["granted"].append(resolved)
            return
        if resolved in _STATE["forbidden"]:
            _STATE["denied"].append(resolved)
            raise PermissionError(
                f"access_guard: opening {C.track_rel(resolved)} is forbidden in this stage"
            )
        if resolved in _STATE["gated"]:
            _STATE["denied"].append(resolved)
            raise PermissionError(
                f"access_guard: {C.track_rel(resolved)} requires a one-shot label grant"
            )


def _ensure_installed() -> None:
    with _LOCK:
        if _STATE["installed"]:
            return
        sys.addaudithook(_hook)
        _STATE["installed"] = True


def arm(gated: Iterable[Path | str] = (C.LIPO_LABELS,), forbidden: Iterable[Path | str] = C.FORBIDDEN_FILES) -> None:
    """Arm the guard for one stage (idempotent; state is reset first)."""
    _ensure_installed()
    with _LOCK:
        _STATE["forbidden"] = {p for p in (_norm(x) for x in forbidden) if p}
        _STATE["gated"] = {p for p in (_norm(x) for x in gated) if p}
        _STATE["grant"] = None
        _STATE["denied"] = []
        _STATE["granted"] = []


def disarm() -> None:
    with _LOCK:
        _STATE["forbidden"] = set()
        _STATE["gated"] = set()
        _STATE["grant"] = None


def granted_files() -> list[str]:
    with _LOCK:
        return [C.track_rel(p) for p in _STATE["granted"]]


def denied_attempts() -> list[str]:
    with _LOCK:
        return [C.track_rel(p) for p in _STATE["denied"]]


def grant_once(path: Path | str) -> None:
    """Allow exactly the next ``open()`` of ``path`` (and no other)."""
    resolved = _norm(path)
    if resolved is None:
        raise ValueError(f"cannot grant a path for {path!r}")
    with _LOCK:
        if _STATE["grant"] is not None:
            raise RuntimeError("a label grant is already pending")
        _STATE["grant"] = resolved


def read_bytes_with_grant(path: Path | str) -> bytes:
    """Read a gated file exactly once as bytes, holding the only grant for it."""
    p = Path(path)
    grant_once(p)
    try:
        with open(p, "rb") as fh:
            return fh.read()
    finally:
        with _LOCK:
            if _STATE["grant"] == _norm(p):
                _STATE["grant"] = None


def read_text_with_grant(path: Path | str, encoding: str = "utf-8") -> str:
    return read_bytes_with_grant(path).decode(encoding)


def guard_status() -> dict:
    with _LOCK:
        return {
            "installed": bool(_STATE["installed"]),
            "armed": bool(_STATE["forbidden"] or _STATE["gated"]),
            "forbidden": [C.track_rel(p) for p in sorted(_STATE["forbidden"])],
            "gated": [C.track_rel(p) for p in sorted(_STATE["gated"])],
            "granted": [C.track_rel(p) for p in _STATE["granted"]],
            "denied_attempts": [C.track_rel(p) for p in _STATE["denied"]],
        }


__all__ = [
    "arm",
    "disarm",
    "grant_once",
    "read_bytes_with_grant",
    "read_text_with_grant",
    "granted_files",
    "denied_attempts",
    "guard_status",
]
