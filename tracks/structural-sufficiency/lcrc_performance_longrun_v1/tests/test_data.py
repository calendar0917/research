"""Data invariants: single/batch consistency, row alignment, one-shot Test view."""

from __future__ import annotations

import numpy as np
import torch

import cl_common as C
import cl_data as CD
import cl_model as CM
import lcrc_data as PREP_DATA
import mb2_data as D


def _predict(model, cache, rows):
    model.eval()
    batch = D.to_device(PREP_DATA.collate(cache, rows), torch.device("cpu"))
    with torch.no_grad():
        return model(batch).detach().cpu().numpy().reshape(len(rows))


def test_single_molecule_equals_batch():
    view = CD.base_view("ESOL", "published")
    model, _ = CM.build_base_model(view.spec, view.train_norm, 0, torch.device("cpu"))
    cache = view.valid_cache
    rows = [0, 1, 2, 3, 4]
    together = _predict(model, cache, rows)
    for i, row in enumerate(rows):
        alone = _predict(model, cache, [row])
        assert abs(float(alone[0]) - float(together[i])) <= 1e-6


def test_rhoshape_single_molecule_equals_batch():
    view = CD.base_view("ESOL", "published")
    model, _ = CM.build_variant_model(
        view.spec, view.spec, view.train_norm, 0, torch.device("cpu"), arm="rhoshape"
    )
    cache = view.valid_cache
    rows = [0, 1, 2]
    together = _predict(model, cache, rows)
    for i, row in enumerate(rows):
        alone = _predict(model, cache, [row])
        assert abs(float(alone[0]) - float(together[i])) <= 1e-6


def test_published_test_rows_align():
    cache, y, rows, base = CD.test_view("ESOL", "published")
    assert [int(x) for x in rows] == [int(x) for x in base.rows["test"]]
    assert len(rows) == int(y.shape[0]) == 113
    assert np.isfinite(y).all()


def test_lipo_repeat_rows_are_frozen():
    view = CD.base_view("Lipophilicity", "lipo-repeat:123")
    frozen = __import__("ta_data").lipo_repeat_split(123)
    assert view.rows["train"] == frozen["rows"]["train"]
    assert view.rows["valid"] == frozen["rows"]["valid"]
    assert len(view.rows["train"]) == 3360 and len(view.rows["valid"]) == 420
    cache, y, rows, _ = CD.test_view("Lipophilicity", "lipo-repeat:123")
    assert rows == frozen["rows"]["test"]
    assert len(rows) == 420 and y.shape == (420, 1)


def test_chem8_view_preserves_rows_and_targets():
    base = CD.base_view("BACE", "published")
    ext = CD.extend_view(base, "BACE")
    assert ext.rows == base.rows
    assert np.array_equal(ext.targets.y_train, base.targets.y_train)
    assert np.array_equal(ext.targets.y_valid, base.targets.y_valid)
    assert ext.target_norm == base.target_norm
