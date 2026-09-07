"""Resolved-config handling: overrides, scientific fingerprinting.

A *scientific config* is the part of a resolved config that determines the
scientific behaviour of a run: it excludes runtime plumbing sections
(``output`` and ``runtime``) and any injected policy keys.  Two runs with the
same runner, scientific config, protocol, seeds and code state are considered
duplicates and are not re-executed without ``--force``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from .serialization import dumps_json, jsonable, sha256_text

# Sections of the config that are plumbing, not science.
_PLUMBING_KEYS = ("output", "runtime")


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge ``override`` into ``base`` (new values win), immutably."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def coerce_scalar(text: str) -> Any:
    """Interpret ``"3" | "0.5" | "true" | "none" | string`` as YAML-like scalar."""
    text = text.strip()
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("none", "null", "~"):
        return None
    # PyYAML-compatible safe parsing for plain scalars.
    try:
        import yaml

        parsed = yaml.safe_load(text)
        if isinstance(parsed, (str, int, float, bool)) or parsed is None:
            if isinstance(parsed, str) and parsed != text:
                raise ValueError(f"ambiguous scalar {text!r}")
            return parsed
    except yaml.YAMLError:
        pass
    raise ValueError(f"cannot interpret override value {text!r}")


def parse_override_spec(spec: str) -> tuple[str, Any]:
    if "=" not in spec:
        raise ValueError(f"--set expects key=value, got {spec!r}")
    key, raw = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"empty override key in {spec!r}")
    return key, coerce_scalar(raw)


def apply_override(config: Mapping[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Apply a dotted-path override (``model.epochs=2``) immutably."""
    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid override key {key!r}")
    root = copy.deepcopy(dict(config))
    cursor: dict[str, Any] = root
    for part in parts[:-1]:
        current = cursor.get(part)
        if current is None:
            current = {}
            cursor[part] = current
        if not isinstance(current, dict):
            raise ValueError(f"cannot descend into non-mapping key {part!r} of {key!r}")
        cursor = current
    cursor[parts[-1]] = copy.deepcopy(value)
    return root


def scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Config without plumbing sections; stable canonical JSON for hashing."""
    return {
        key: copy.deepcopy(value)
        for key, value in config.items()
        if key not in _PLUMBING_KEYS
    }


def config_hash(config: Mapping[str, Any]) -> str:
    return sha256_text(dumps_json(jsonable(scientific_config(config))))


def load_base_config(path: Path) -> dict[str, Any]:
    """Load a YAML config and confirm it is a mapping."""
    from .serialization import load_yaml

    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"config {path} does not contain a mapping")
    return payload
