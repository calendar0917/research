"""Fast, CPU-only tests for the research-integrity validator.

These never import torch, never touch data and never run an experiment; they
are safe for CI and for the local fast subset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ksvd_research.runtime import integrity


def _run_repo_checks():
    return integrity.integrity_checks()


def test_repo_integrity_passes():
    checks = _run_repo_checks()
    errors = [c for c in checks if not c.ok and c.severity == "error"]
    assert not errors, [c.detail for c in errors]
    assert integrity.check_exit_code(checks) == 0


def test_repo_documented_future_dates_still_ok():
    """The 16 documented future-dated records stay allowlisted (no regression)."""
    checks = _run_repo_checks()
    future = [c for c in checks if c.name == "records.no_undocumented_future_dates"][0]
    assert future.ok, future.detail


def test_repo_run_record_pointers_resolve():
    checks = _run_repo_checks()
    runs = [c for c in checks if c.name == "records.run_pointers_resolve"][0]
    assert runs.ok, runs.detail


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fake_tree(tmp_path: Path, *, records):
    """Create a minimal STATE/records tree; returns the records root."""
    records_root = tmp_path / "records"
    _write(
        records_root / "claims" / "claim-keep-20260101.yaml",
        "claim_id: claim-keep-20260101\ncreated: '2026-01-01T00:00:00'\nstatus: supported\n",
    )
    for name, body in records.items():
        kind = "claims" if name.startswith("claim-") else "decisions"
        _write(records_root / kind / f"{name}.yaml", body)
    _write(
        tmp_path / "STATE.yaml",
        "\n".join(
            [
                "phase: test",
                "active_studies: [zinc-context-gap]",
                "focus_study: zinc-context-gap",
                "authorized_next_action: {status: none}",
                "guardrails: ['keep']",
            ]
        ),
    )
    return records_root


@pytest.fixture
def patched(tmp_path, monkeypatch):
    records_root = _fake_tree(tmp_path, records={})
    monkeypatch.setattr(integrity, "records_root", lambda: records_root)
    monkeypatch.setattr(integrity, "state_file", lambda: tmp_path / "STATE.yaml")
    monkeypatch.setattr(integrity, "track_root", lambda: tmp_path)
    return tmp_path


def test_undocumented_future_date_is_error(patched, monkeypatch):
    from datetime import date

    _write(
        patched / "records" / "claims" / "claim-future-20991231.yaml",
        "claim_id: claim-future-20991231\ncreated: '2099-12-31T00:00:00'\nstatus: supported\n",
    )
    checks = integrity.integrity_checks(today=date(2026, 9, 18))
    future = [c for c in checks if c.name == "records.no_undocumented_future_dates"][0]
    assert not future.ok
    assert "claim-future-20991231" in future.detail
    assert integrity.check_exit_code(checks) == 1


def test_documented_future_date_is_not_an_error(patched, monkeypatch):
    from datetime import date

    _write(
        patched / "records" / "claims" / "claim-future-20260920.yaml",
        "claim_id: claim-future-20260920\ncreated: '2026-09-20T00:00:00'\nstatus: supported\n",
    )
    _write(
        patched / "records" / "PROVENANCE_DATES.md",
        "```yaml\ndocumented_future_dated_records:\n  - record: claim-future-20260920\n```\n",
    )
    checks = integrity.integrity_checks(today=date(2026, 9, 18))
    future = [c for c in checks if c.name == "records.no_undocumented_future_dates"][0]
    assert future.ok


def test_dangling_reference_is_error(patched, monkeypatch):
    _write(
        patched / "STATE.yaml",
        "\n".join(
            [
                "phase: test",
                "active_studies: [zinc-context-gap]",
                "focus_study: zinc-context-gap",
                "authorized_next_action: {status: none}",
                "guardrails: ['see decision-does-not-exist-20260101']",
            ]
        ),
    )
    checks = integrity.integrity_checks()
    refs = [c for c in checks if c.name == "records.references_resolve"][0]
    assert not refs.ok
    assert "decision-does-not-exist-20260101" in refs.detail


def test_dangling_run_record_pointer_is_error(patched):
    _write(
        patched / "STATE.yaml",
        "\n".join(
            [
                "phase: test",
                "active_studies: [zinc-context-gap]",
                "focus_study: zinc-context-gap",
                "authorized_next_action: {status: none}",
                "guardrails: ['keep']",
                "baseline_record: record-missing-20260101",
            ]
        ),
    )
    checks = integrity.integrity_checks()
    runs = [c for c in checks if c.name == "records.run_pointers_resolve"][0]
    assert not runs.ok
    assert "record-missing-20260101" in runs.detail
    assert integrity.check_exit_code(checks) == 1


def test_valid_run_record_pointer_passes(patched):
    _write(
        patched / "records" / "runs" / "record-real-20260101.json",
        '{"record_id": "record-real-20260101"}\n',
    )
    _write(
        patched / "STATE.yaml",
        "\n".join(
            [
                "phase: test",
                "active_studies: [zinc-context-gap]",
                "focus_study: zinc-context-gap",
                "authorized_next_action: {status: none}",
                "guardrails: ['keep']",
                "baseline_record: record-real-20260101",
            ]
        ),
    )
    checks = integrity.integrity_checks()
    runs = [c for c in checks if c.name == "records.run_pointers_resolve"][0]
    assert runs.ok, runs.detail


def test_repository_timezone_crosses_utc_midnight():
    from datetime import date, datetime, timezone

    # 16:30 UTC == 00:30 the next calendar day in Asia/Shanghai (UTC+8).
    assert integrity.repository_today(
        datetime(2026, 9, 18, 16, 30, tzinfo=timezone.utc)
    ) == date(2026, 9, 19)
    # 15:30 UTC == 23:30 the same day in Asia/Shanghai.
    assert integrity.repository_today(
        datetime(2026, 9, 18, 15, 30, tzinfo=timezone.utc)
    ) == date(2026, 9, 18)
    # A naive timestamp is interpreted as already repository-local.
    assert integrity.repository_today(datetime(2026, 9, 19, 0, 30)) == date(2026, 9, 19)


def test_local_evening_record_is_not_future_dated(patched):
    from datetime import datetime, timezone

    _write(
        patched / "records" / "claims" / "claim-local-20260919.yaml",
        "claim_id: claim-local-20260919\ncreated: '2026-09-19T00:30:00'\nstatus: supported\n",
    )
    # At 2026-09-18 16:30 UTC the repository-local date is already 2026-09-19,
    # so a record written at 00:30 local must not be flagged as future-dated.
    today = integrity.repository_today(datetime(2026, 9, 18, 16, 30, tzinfo=timezone.utc))
    checks = integrity.integrity_checks(today=today)
    future = [c for c in checks if c.name == "records.no_undocumented_future_dates"][0]
    assert future.ok, future.detail


def test_id_filename_mismatch_is_error(patched, monkeypatch):
    _write(
        patched / "records" / "claims" / "claim-abc-20260101.yaml",
        "claim_id: claim-wrong-20260101\ncreated: '2026-01-01T00:00:00'\nstatus: supported\n",
    )
    checks = integrity.integrity_checks()
    ids = [c for c in checks if c.name == "records.id_matches_filename"][0]
    assert not ids.ok


def test_cli_verify_json(monkeypatch, capsys):
    import argparse

    from ksvd_research import cli

    rc = cli._cmd_verify(argparse.Namespace(json=True))
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload
    assert rc == 0
