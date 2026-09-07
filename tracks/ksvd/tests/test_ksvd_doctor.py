"""Doctor behavior: a fresh clone without local runs/ must not fail.

The scratch run store is lazy; the git-tracked records tree and the control
files are the ground truth.  Package checks split required (FAIL) from
optional (WARN), and only errors produce a non-zero exit code.
"""

from __future__ import annotations

from ksvd_research.cli import doctor_checks, doctor_exit_code
from ksvd_research.runtime.environment import capture_environment


def test_doctor_allows_missing_local_run_store(tmp_path, monkeypatch):
    import ksvd_research.cli as cli

    missing = tmp_path / "runs"
    monkeypatch.setattr(cli, "runs_root", lambda: missing)
    checks = doctor_checks()
    runs_check = next((check for check in checks if check[0] == "control.runs"), None)
    assert runs_check is not None
    assert runs_check[1] is True, f"missing runs/ must not fail doctor: {runs_check}"
    assert runs_check[3] == "ok"
    assert "lazily" in runs_check[2]
    assert "not created yet" in runs_check[2]


def test_doctor_checks_normal_configuration_has_no_error():
    checks = doctor_checks()
    errors = [check for check in checks if check[3] == "error"]
    assert errors == [], f"doctor should be green in this repo: {errors}"


def test_doctor_exit_code_warn_is_zero():
    checks = [
        ("a", True, "x", "ok"),
        ("b", False, "optional absent", "warn"),
    ]
    assert doctor_exit_code(checks) == 0


def test_doctor_exit_code_error_is_nonzero():
    checks = [
        ("a", True, "x", "ok"),
        ("b", False, "missing required", "error"),
    ]
    assert doctor_exit_code(checks) != 0


def test_environment_records_missing_package_as_null():
    environment = capture_environment()
    packages = environment["packages"]
    for name in (
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "pyyaml",
        "torch",
        "torch-geometric",
        "ogb",
        "networkx",
        "pynauty",
        "optuna",
        "pytest",
    ):
        assert name in packages, f"package key {name} must exist (value may be null)"
    assert packages["numpy"] is not None
    for key in ("python", "python_full", "executable", "platform", "machine", "cwd"):
        assert key in environment
