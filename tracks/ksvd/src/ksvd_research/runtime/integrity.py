"""Lightweight, deterministic research-integrity checks.

This module checks the *durable* research surface — ``STATE.yaml`` and the
``records/`` tree — without importing torch, touching data, or running any
experiment.  It is meant to be safe for CI: pure-python + PyYAML only.

Checks performed
----------------
* STATE.yaml loads and declares the keys the control plane needs.
* Every claim/decision YAML loads as a mapping.
* Record id (filename stem) matches the ``claim_id`` / ``decision_id`` field.
* The date embedded in a record filename is consistent with its ``created``
  timestamp and with the date embedded in its id.
* Timestamps that are in the *future* relative to ``--today`` are reported.
  A record whose id appears in the documented allowlist
  (``records/PROVENANCE_DATES.md``) is reported as ``documented``; an
  undocumented future date is an ``error``.
* Every claim/decision id mentioned in STATE.yaml resolves to a record file.
* Cross-references of the form ``records/<kind>/<id>.yaml`` and
  ``notes/<name>.md`` written inside records exist on disk.

Nothing here selects models, reads the official test, or changes scientific
content; it only reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from .paths import records_root, state_file, track_root
from .serialization import load_yaml

CLAIM_ID_RE = re.compile(r"\bclaim-[A-Za-z0-9][A-Za-z0-9-]*\b")
DECISION_ID_RE = re.compile(r"\bdecision-[A-Za-z0-9][A-Za-z0-9-]*\b")
_FILENAME_DATE_RE = re.compile(r"-(\d{8})$")
_DATE_IN_ID_RE = re.compile(r"(\d{8})")
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")

STATE_REQUIRED_KEYS = (
    "phase",
    "active_studies",
    "focus_study",
    "open_questions",
    "guardrails",
)

PROVENANCE_FILE = "PROVENANCE_DATES.md"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    severity: str = "ok"  # ok | warn | error | documented


def _severity(ok: bool, severity: str) -> str:
    if ok:
        return "ok" if severity in ("ok", "documented") else severity
    return severity if severity in ("warn", "error") else "error"


def _parse_date_token(value: str) -> date | None:
    value = value.strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _record_date_from_stem(stem: str) -> date | None:
    match = _FILENAME_DATE_RE.search(stem)
    if not match:
        return None
    return _parse_date_token(match.group(1))


def _created_date(payload: dict[str, Any]) -> date | None:
    raw = payload.get("created")
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        match = _ISO_DATE_RE.search(raw)
        if match:
            return _parse_date_token(match.group(1))
    return None


def _id_date(identifier: str) -> date | None:
    match = _DATE_IN_ID_RE.search(identifier)
    if not match:
        return None
    return _parse_date_token(match.group(1))


def _documented_future_dates() -> set[str]:
    """Read the machine list of documented future dates from PROVENANCE_DATES.md."""
    path = records_root() / PROVENANCE_FILE
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8")
    block = None
    if "```yaml" in text:
        frag = text.split("```yaml", 1)[1].split("```", 1)[0]
        block = frag
    if block is None:
        return set()
    try:
        payload = yaml.safe_load(block)
    except Exception:  # noqa: BLE001 - provenance is advisory metadata
        return set()
    documented: set[str] = set()
    for entry in (payload or {}).get("documented_future_dated_records", []) or []:
        if isinstance(entry, dict) and entry.get("record"):
            documented.add(str(entry["record"]))
    return documented


def _iter_records(kind: str) -> list[Path]:
    root = records_root() / kind
    if not root.is_dir():
        return []
    return sorted(root.glob("*.yaml"))


def _collect_ids(pattern: re.Pattern[str], *texts: str) -> set[str]:
    found: set[str] = set()
    for text in texts:
        found.update(_filter_ids(pattern.findall(text or "")))
    return found


_RECORD_ID_SUFFIX_RE = re.compile(r"(?:-\d{8}|-[0-9a-f]{8})$")


def _filter_ids(candidates: list[str]) -> set[str]:
    """Keep only tokens that actually look like a claim/decision id.

    English phrases such as "decision-tree" or "decision-relevant" match the
    id prefix but are not records; real ids always end in a date or an 8-hex
    suffix.
    """
    return {c for c in candidates if _RECORD_ID_SUFFIX_RE.search(c)}


def _record_body_ids(payload: dict[str, Any]) -> list[str]:
    """Claim/decision ids embedded inside a record body (array or scalar fields)."""
    ids: list[str] = []
    for key in ("supporting_evidence", "contradicting_evidence", "limitations",
                "consequences", "alternatives_rejected", "evidence", "scope",
                "statement", "decision", "because", "revisit_if"):
        value = payload.get(key)
        if isinstance(value, str):
            ids.extend(_filter_ids(CLAIM_ID_RE.findall(value)))
            ids.extend(_filter_ids(DECISION_ID_RE.findall(value)))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    ids.extend(_filter_ids(CLAIM_ID_RE.findall(item)))
                    ids.extend(_filter_ids(DECISION_ID_RE.findall(item)))
                elif isinstance(item, dict):
                    for sub in item.values():
                        if isinstance(sub, str):
                            ids.extend(_filter_ids(CLAIM_ID_RE.findall(sub)))
                            ids.extend(_filter_ids(DECISION_ID_RE.findall(sub)))
    return ids


_PATH_RE = re.compile(r"(?:records/(?:claims|decisions|runs)/[A-Za-z0-9._-]+|notes/[A-Za-z0-9._-]+\.md)")


def integrity_checks(today: date | None = None) -> list[Check]:
    """Run all integrity checks and return an ordered list of results."""
    today = today or date.today()
    checks: list[Check] = []

    # --- STATE ---------------------------------------------------------
    state: dict[str, Any] = {}
    try:
        payload = load_yaml(state_file())
        if not isinstance(payload, dict):
            raise ValueError("STATE.yaml is not a mapping")
        state = payload
        checks.append(Check("state.load", True, f"{state_file().name} parsed"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("state.load", False, repr(exc), "error"))

    if state:
        missing = [key for key in STATE_REQUIRED_KEYS if key not in state]
        checks.append(
            Check(
                "state.required_keys",
                not missing,
                "present" if not missing else f"MISSING: {', '.join(missing)}",
            )
        )
        empty = [
            key
            for key in ("open_questions", "guardrails", "active_studies")
            if not state.get(key)
        ]
        checks.append(
            Check("state.non_empty", not empty, "non-empty" if not empty else f"empty: {empty}")
        )

    # --- record loading / id / date consistency ------------------------
    documented = _documented_future_dates()
    all_ids: set[str] = set()
    bodies: list[tuple[str, Path, dict[str, Any]]] = []
    load_errors: list[str] = []
    id_errors: list[str] = []
    date_errors: list[str] = []
    future_undocumented: list[str] = []
    future_documented: list[str] = []
    duplicates: list[str] = []

    for kind, id_field, id_re in (
        ("claims", "claim_id", CLAIM_ID_RE),
        ("decisions", "decision_id", DECISION_ID_RE),
    ):
        for path in _iter_records(kind):
            try:
                payload = load_yaml(path)
            except Exception as exc:  # noqa: BLE001
                load_errors.append(f"{path.name}: {exc!r}")
                continue
            if not isinstance(payload, dict):
                load_errors.append(f"{path.name}: not a mapping")
                continue
            bodies.append((kind, path, payload))
            stem = path.stem
            field_value = payload.get(id_field)
            if field_value != stem:
                id_errors.append(f"{path.name}: {id_field}={field_value!r}")
            if stem in all_ids:
                duplicates.append(stem)
            all_ids.add(stem)

            file_date = _record_date_from_stem(stem)
            created = _created_date(payload)
            inline_id = _id_date(str(field_value or stem))
            dates = [d for d in (file_date, created, inline_id) if d]
            if file_date and created and file_date != created:
                date_errors.append(
                    f"{path.name}: filename {file_date} != created {created}"
                )
            if file_date and inline_id and file_date != inline_id:
                date_errors.append(
                    f"{path.name}: filename {file_date} != id date {inline_id}"
                )
            if any(d > today for d in dates):
                if stem in documented:
                    future_documented.append(stem)
                else:
                    future_undocumented.append(stem)

    checks.append(Check("records.load", not load_errors, f"{len(bodies)} records parsed"
                        if not load_errors else "; ".join(load_errors[:6])))
    checks.append(
        Check("records.id_matches_filename", not id_errors,
              "all ids match" if not id_errors else "; ".join(id_errors[:6]))
    )
    checks.append(
        Check("records.no_duplicate_ids", not duplicates,
              "unique" if not duplicates else f"duplicates: {duplicates}")
    )
    checks.append(
        Check("records.date_internal_consistency", not date_errors,
              "consistent" if not date_errors else "; ".join(date_errors[:6]),
              severity="warn")
    )
    if future_undocumented:
        checks.append(
            Check(
                "records.no_undocumented_future_dates",
                False,
                "undocumented future-dated records: " + ", ".join(future_undocumented[:12]),
                "error",
            )
        )
    else:
        checks.append(
            Check(
                "records.no_undocumented_future_dates",
                True,
                f"documented future-dated records: {len(future_documented)} (see records/{PROVENANCE_FILE})",
            )
        )

    # --- reference resolution ------------------------------------------
    state_text = state_file().read_text(encoding="utf-8") if state_file().is_file() else ""
    referenced = _collect_ids(CLAIM_ID_RE, state_text) | _collect_ids(DECISION_ID_RE, state_text)
    # ids written inside record bodies must exist too
    for _kind, _path, payload in bodies:
        referenced.update(_record_body_ids(payload))
    dangling = sorted(
        rid for rid in referenced
        if rid not in all_ids and not (records_root() / "claims" / f"{rid}.yaml").is_file()
        and not (records_root() / "decisions" / f"{rid}.yaml").is_file()
    )
    checks.append(
        Check(
            "records.references_resolve",
            not dangling,
            "all claim/decision references resolve"
            if not dangling
            else f"dangling: {', '.join(dangling[:12])}",
        )
    )

    # --- path references inside records / STATE ------------------------
    root = track_root()
    missing_paths: list[str] = []
    for text in [state_text] + [
        " ".join(str(v) for v in payload.values() if isinstance(v, str))
        for _k, _p, payload in bodies
    ]:
        for rel in _PATH_RE.findall(text or ""):
            if not (root / rel).is_file():
                missing_paths.append(rel)
    missing_paths = sorted(set(missing_paths))
    checks.append(
        Check(
            "records.path_references_exist",
            not missing_paths,
            "all note/record path references exist"
            if not missing_paths
            else f"missing: {', '.join(missing_paths[:12])}",
            severity="warn",
        )
    )

    return checks


def check_exit_code(checks: list[Check]) -> int:
    return 1 if any(c.severity == "error" and not c.ok for c in checks) else 0
