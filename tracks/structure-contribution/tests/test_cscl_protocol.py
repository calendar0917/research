"""Protocol-level tests for CSCL-v0: split discipline, label isolation,
official-test embargo, data-path integration (skipped when data absent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402

HAVE_ZINC = (Path(__file__).resolve().parents[3] / "data/ZINC/subset/processed/train.pt").exists()


def test_split_sizes_and_disjointness():
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    assert len(smiles) == 10000
    idx = cf.build_split_indices(smiles)
    fit, dev = set(idx["fit_all"].tolist()), set(idx["dev"].tolist())
    assert len(fit) == 8000 and len(dev) == 2000 and not (fit & dev)
    inner, mon = set(idx["fit_inner"].tolist()), set(idx["monitor"].tolist())
    assert len(inner) == 7200 and len(mon) == 800
    assert not (inner & mon) and (inner | mon) == fit


def test_split_is_deterministic():
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    a = cf.build_split_indices(smiles)
    b = cf.build_split_indices(smiles)
    for key in ("fit_inner", "monitor", "dev", "fit_all"):
        assert np.array_equal(a[key], b[key]), key


def test_no_duplicate_smiles_crosses_splits():
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    idx = cf.build_split_indices(smiles)
    for left, right in (("fit_inner", "monitor"), ("fit_all", "dev")):
        ls = {smiles[i] for i in idx[left]}
        rs = {smiles[i] for i in idx[right]}
        assert not (ls & rs), f"duplicate molecule crosses {left}/{right}"


def test_official_test_is_never_loaded(monkeypatch):
    """The loader must request the official **train** split only."""
    calls: list[str] = []

    def fake_loader(root, split):
        calls.append(split)
        return []

    import tracks.ksvd.experiments.luyin16.zinc_long_range_proxy as proxy

    monkeypatch.setattr(proxy, "_load_zinc", fake_loader)
    cf.load_official_train()
    assert calls == ["train"]
    assert "test" not in calls and "val" not in calls


def test_only_raw_y_enters_supervision():
    y = np.array([[0.5], [-2.0], [7.25]], dtype=np.float32)
    t = cf.y_only_targets(y)
    assert np.array_equal(t, y[:, 0])


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_ZINC, reason="data/ZINC not present")
def test_prepare_data_integration():
    """Full extraction on the real data + audit trigger statistics."""
    from run_cscl_v0 import prepare_data

    data = prepare_data()
    stats = data["stats"]
    assert stats.vocab.n_total > 0
    # supervision shape: raw y only
    assert data["y"].shape == (10000, 1)
    # every molecule has tensors and relations reference valid unit indices
    for ts, mols in ((data["fit_inner"], data["fit_inner_mols"]), (data["dev"], data["dev_mols"])):
        for t, m in zip(ts, mols):
            k = len(m.unit_sigs)
            assert len(t.type_ids) == k and t.desc.shape == (k, cf.DESC_DIM)
            n_rel = len(m.rel_pairs)
            assert len(t.rel_feat) == n_rel
            if n_rel:
                assert t.rel_unit_lo.max() < k and t.rel_unit_hi.max() < k
    # fit-only vocab: dev unknown signatures route to UNK ids (no crash)
    dev_ids = [t.type_ids.max() for t in data["dev"]]
    assert max(dev_ids) < stats.vocab.n_total + 1
