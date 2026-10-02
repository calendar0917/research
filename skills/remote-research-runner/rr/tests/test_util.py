"""Unit tests for shared utilities."""

from __future__ import annotations

from rr.util import (
    b64,
    driver_at_least,
    expected_cuda_for_backend,
    make_run_id,
    normalize_slurm_state,
    parse_driver,
    shell_export_lines,
    valid_run_id,
)


def test_parse_driver():
    assert parse_driver("525.85.12") == (525, 85, 12)
    assert parse_driver("510.47.03") == (510, 47, 3)
    assert parse_driver("550.163") == (550, 163, 0)
    assert parse_driver("") == (0, 0, 0)


def test_driver_at_least():
    assert driver_at_least("525.85.12", "525.60.13")
    assert driver_at_least("550.163.01", "525.60.13")
    assert not driver_at_least("510.108.03", "525.60.13")
    assert not driver_at_least(None, "525.60.13")
    assert driver_at_least(None, None)
    assert driver_at_least("510.0.0", None)


def test_run_id_shape():
    rid = make_run_id("zinc-seed0")
    assert rid.startswith("zinc-seed0-")
    assert valid_run_id(rid)
    assert not valid_run_id("has space")
    assert not valid_run_id("-leading")


def test_shell_export_lines_quotes():
    text = shell_export_lines({"A": "plain", "B": "has space", "C": "x'y", "D": None})
    assert "export A=plain" in text
    assert "export B='has space'" in text
    assert "export C=" in text
    assert "D" not in text


def test_b64_roundtrip():
    import base64

    payload = "python -m x --a 1"
    assert base64.b64decode(b64(payload)).decode() == payload


# ---------------------------------------------------------------------------
# Slurm state normalization
# ---------------------------------------------------------------------------

def test_normalize_slurm_state_cancelled_decorations():
    # sacct reports cancelled jobs in several decorated forms.
    assert normalize_slurm_state("CANCELLED+") == "CANCELLED"
    assert normalize_slurm_state("CANCELLED by 12345") == "CANCELLED"
    assert normalize_slurm_state("CANCELLED by somebody") == "CANCELLED"
    assert normalize_slurm_state("cancelled by 99") == "CANCELLED"


def test_normalize_slurm_state_terminal_and_live():
    assert normalize_slurm_state("OUT_OF_MEMORY") == "OUT_OF_MEMORY"
    assert normalize_slurm_state("TIMEOUT") == "TIMEOUT"
    assert normalize_slurm_state("NODE_FAIL") == "NODE_FAIL"
    assert normalize_slurm_state("PREEMPTED") == "PREEMPTED"
    assert normalize_slurm_state("COMPLETED") == "COMPLETED"
    assert normalize_slurm_state("RUNNING") == "RUNNING"
    assert normalize_slurm_state("PENDING") == "PENDING"


def test_normalize_slurm_state_edges():
    assert normalize_slurm_state("") == ""
    assert normalize_slurm_state(None) == ""
    assert normalize_slurm_state(" Out_Of_Memory ") == "OUT_OF_MEMORY"


def test_expected_cuda_for_backend():
    assert expected_cuda_for_backend("cu118") == "11.8"
    assert expected_cuda_for_backend("cu121") == "12.1"
    assert expected_cuda_for_backend("cu124") == "12.4"
    assert expected_cuda_for_backend("cu128") == "12.8"
    assert expected_cuda_for_backend("cpu") is None
    assert expected_cuda_for_backend(None) is None