from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

import c2_common as C
import c2_data as CD
import c2_models as CM

import r42_data as D


def _train_degree_tmp(tmp: Path) -> None:
    split = D.load_split()
    data = D.load_structure_data(rows=split["rows"]["train"])
    deg = CM.pna_degree_histogram(data)
    (tmp / "saved_PNA_degrees").mkdir(parents=True, exist_ok=True)
    torch.save(deg["tensor"], tmp / "saved_PNA_degrees" / "degrees.pt")


def test_selected_width_45_parameter_count():
    dims = CD.structure_dimensions()
    with tempfile.TemporaryDirectory() as name:
        tmp = Path(name)
        _train_degree_tmp(tmp)
        model = CM.pna_estimator(width=45, num_features=dims["num_features"], edge_dim=dims["edge_dim"], out_path=tmp)
        params = sum(p.numel() for p in model.parameters())
    assert params == 256_041
    assert abs(params - C.PNA_TARGET_PARAMS) / C.PNA_TARGET_PARAMS < C.PNA_SIZE_TOLERANCE


@pytest.mark.skipif(not C.WIDTH_SEARCH.exists(), reason="width search has not been run yet")
def test_width_search_payload_is_label_free_and_frozen():
    payload = C.read_json(C.WIDTH_SEARCH)
    assert payload["labels_read"] is False
    assert payload["validation_scores_read"] is False
    assert payload["selected_width"] == 45
    assert payload["selected_parameters_total"] == 256_041
    assert payload["tightly_size_matched"] is True
    best = min(payload["candidates"], key=lambda row: (row["absolute_difference"], row["width"]))
    assert best["width"] == payload["selected_width"]
    assert best["parameters_total"] == payload["selected_parameters_total"]
