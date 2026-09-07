"""Protocol / study / STATE loading for the control plane.

Protocols encode comparability rules (dataset, split, metric, fitting scope,
seed semantics); studies encode scientific intent.  STATE.yaml is the small
live pointer file the AI reads to orient itself quickly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import protocols_root, state_file, studies_root
from .serialization import dumps_json, jsonable, load_yaml, sha256_text


def load_protocol(protocol_id: str) -> dict[str, Any]:
    path = protocols_root() / f"{protocol_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no protocol named {protocol_id!r} at {path}")
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"protocol {path} is not a mapping")
    return payload


def protocol_hash(protocol: dict[str, Any]) -> str:
    """Semantic hash of a parsed protocol.

    Hashes the parsed content only: YAML formatting, comments and key
    ordering must not change the hash; a change to any scientific rule does.
    """
    if not isinstance(protocol, dict):
        raise ValueError("protocol hash expects a mapping")
    return sha256_text(dumps_json(jsonable(protocol)))


def load_study(study_id: str) -> dict[str, Any]:
    path = studies_root() / f"{study_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no study named {study_id!r} at {path}")
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"study {path} is not a mapping")
    return payload


def load_state() -> dict[str, Any]:
    path = state_file()
    if not path.is_file():
        raise FileNotFoundError(f"STATE.yaml not found at {path}")
    payload = load_yaml(path)
    if not isinstance(payload, dict):
        raise ValueError(f"STATE.yaml at {path} is not a mapping")
    return payload


def protocol_for_study(study_id: str) -> dict[str, Any]:
    study = load_study(study_id)
    protocol_ref = study.get("protocol")
    if not protocol_ref:
        raise ValueError(f"study {study_id!r} does not declare a protocol")
    protocol_path = Path(str(protocol_ref))
    if str(protocol_ref).endswith(".yaml"):
        protocol_id = protocol_path.stem
    else:
        protocol_id = str(protocol_ref)
    return load_protocol(protocol_id)
