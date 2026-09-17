from __future__ import annotations

import json

import pytest

from tracks.ksvd.experiments.luyin16 import (
    molhiv_local_token_null_test_closure as mtc,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_token_null_test_closure as ztc,
)


def test_zinc_expected_totals() -> None:
    assert ztc.EXPECTED_TOTAL == {"null": 49343, "constant": 49359}
    assert [rep for rep, _ in ztc.CANDIDATES] == ["null", "constant"]


def test_molhiv_expected_totals() -> None:
    assert mtc.CANDIDATES == ("null", "constant")
    assert mtc._expected_total("null", 26233, 103) == 237133
    assert mtc._expected_total("constant", 26233, 103) == 237165


def test_zinc_refuses_test_without_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ztc, "RESULTS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="architecture_freeze.json missing"):
        ztc.test_eval()


def test_molhiv_refuses_test_without_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mtc, "RESULTS_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="architecture_freeze.json missing"):
        mtc.test_eval(device="cpu")


def test_zinc_refuses_second_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ztc, "RESULTS_DIR", tmp_path)
    (tmp_path / "architecture_freeze.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing architecture freeze"):
        ztc.freeze()


def test_molhiv_refuses_second_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mtc, "RESULTS_DIR", tmp_path)
    (tmp_path / "architecture_freeze.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="existing architecture freeze"):
        mtc.freeze()


def test_zinc_refuses_test_after_unlock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ztc, "RESULTS_DIR", tmp_path)
    (tmp_path / "architecture_freeze.json").write_text(
        json.dumps({"test_loaded_at_freeze_time": False}), encoding="utf-8"
    )
    (tmp_path / "official_test_unlock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already unlocked once"):
        ztc.test_eval()


def test_molhiv_refuses_test_after_unlock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(mtc, "RESULTS_DIR", tmp_path)
    (tmp_path / "architecture_freeze.json").write_text(
        json.dumps({"test_loaded_at_freeze_time": False}), encoding="utf-8"
    )
    (tmp_path / "official_test_unlock.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already unlocked once"):
        mtc.test_eval(device="cpu")
