"""Claim and decision records.

Claims capture tentative/supported scientific statements with evidence;
decisions capture rationale with alternatives and revisit conditions.  They
are small YAML files in the git-tracked ``records/`` tree.
"""

from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import records_root
from .serialization import load_yaml, write_yaml_atomic

CLAIM_STATUSES = ("tentative", "supported", "refuted", "superseded")


def _yaml_path(root: Path, identifier: str) -> Path:
    file_name = identifier
    if not file_name.endswith((".yaml", ".yml")):
        file_name = f"{file_name}.yaml"
    return root / file_name


def new_claim(
    *,
    statement: str,
    study_id: str | None = None,
    status: str = "tentative",
    scope: str | None = None,
    supporting_evidence: list[str] | None = None,
    contradicting_evidence: list[str] | None = None,
    limitations: list[str] | None = None,
    implication: str | None = None,
    claim_id: str | None = None,
) -> dict[str, Any]:
    if status not in CLAIM_STATUSES:
        raise ValueError(f"invalid claim status {status!r}; expected one of {CLAIM_STATUSES}")
    if not statement.strip():
        raise ValueError("claim statement is required")
    record = {
        "kind": "claim",
        "claim_id": claim_id or f"claim-{secrets.token_hex(4)}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "statement": statement.strip(),
        "status": status,
        "scope": scope,
        "study_id": study_id,
        "supporting_evidence": supporting_evidence or [],
        "contradicting_evidence": contradicting_evidence or [],
        "limitations": limitations or [],
        "implication": implication,
    }
    path = _yaml_path(records_root() / "claims", record["claim_id"])
    if path.is_file():
        existing = load_yaml(path)
        existing.pop("updated", None)
        incoming = dict(record)
        incoming.pop("updated", None)
        if existing != incoming:
            raise RuntimeError(f"claim {path} already exists with different content")
        return record
    write_yaml_atomic(path, record)
    return record


def set_claim_status(claim_id: str, status: str) -> dict[str, Any]:
    if status not in CLAIM_STATUSES:
        raise ValueError(f"invalid claim status {status!r}")
    path = _yaml_path(records_root() / "claims", claim_id)
    if not path.is_file():
        raise FileNotFoundError(f"no claim {claim_id!r}")
    record = load_yaml(path)
    record["status"] = status
    record["updated"] = datetime.now().isoformat(timespec="seconds")
    write_yaml_atomic(path, record)
    return record


def list_claims() -> list[tuple[str, Path]]:
    root = records_root() / "claims"
    if not root.is_dir():
        return []
    return sorted(
        (path.stem, path) for path in root.glob("*.yaml")
    )


def new_decision(
    *,
    decision: str,
    because: str,
    alternatives_rejected: list[str] | None = None,
    revisit_if: str | None = None,
    study_id: str | None = None,
    decision_id: str | None = None,
) -> dict[str, Any]:
    if not decision.strip() or not because.strip():
        raise ValueError("decision and because are required")
    record = {
        "kind": "decision",
        "decision_id": decision_id or f"decision-{secrets.token_hex(4)}",
        "created": datetime.now().isoformat(timespec="seconds"),
        "decision": decision.strip(),
        "because": because.strip(),
        "alternatives_rejected": alternatives_rejected or [],
        "revisit_if": revisit_if,
        "study_id": study_id,
    }
    path = _yaml_path(records_root() / "decisions", record["decision_id"])
    if path.is_file():
        existing = load_yaml(path)
        existing.pop("updated", None)
        incoming = dict(record)
        incoming.pop("updated", None)
        if existing != incoming:
            raise RuntimeError(f"decision {path} already exists with different content")
        return record
    write_yaml_atomic(path, record)
    return record


def list_decisions() -> list[tuple[str, Path]]:
    root = records_root() / "decisions"
    if not root.is_dir():
        return []
    return sorted(
        (path.stem, path) for path in root.glob("*.yaml")
    )
