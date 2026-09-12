"""Tests for the compact-v4 parameter allocation & leverage audit.

These tests pin the protocol discipline and measurement integrity of the
analysis-only audit; no backbone is trained and official test is never loaded:

1.  parameter ledger sums to 99,613 and does not double count tensors;
2.  the parameter class summary is consistent with the ledger;
3.  forward-hook instrumentation does not change any prediction;
4.  token frequency is computed from official-train inputs only;
5.  frequency/spectral analysis never reads the target;
6.  SVD rank cutoffs are mechanically derived from 99% / 99.9% energy;
7.  low-rank perturbation changes only the lookup, never other parameters;
8.  the two optimized checkpoint fingerprints are recomputed correctly;
9.  the pre-head representation R is 302D;
10. the small head is fixed at 302->13->13->1 (4135) with no search;
11. the small-head fit/selection split is the official-train internal split;
12. official valid never participates in small-head checkpoint selection;
13. official test is never loaded;
14. activation instrumentation does not change any prediction;
15. the gradient proxy never updates a parameter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import (
    parl_parameter_leverage_audit as pa,
)

RESULTS = pa.RESULTS_DIR


# ---------------------------------------------------------------------------
# synthetic fixtures / lightweight caches
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _encoded():
    if "enc" not in _CACHE:
        _CACHE["enc"] = pa.bss._build_encoded()
    return _CACHE["enc"]


def _model(seed: int = 0):
    key = f"model{seed}"
    if key not in _CACHE:
        _CACHE[key] = pa.bss._load_optimized_model(seed)[0]
    return _CACHE[key]


# ---------------------------------------------------------------------------
# 1-2  ledger
# ---------------------------------------------------------------------------

def test_parameter_ledger_sums_to_99613_and_unique():
    df = pa.gather_ledger()
    assert int(df["params"].sum()) == 99613
    assert int(df["params"].sum()) == pa.EXPECTED_PARAMS
    assert df["tensor/module"].is_unique


def test_parameter_class_summary_consistent():
    df = pa.gather_ledger()
    summary = pa._class_summary(df)
    assert summary["total_params"] == 99613
    parts = sum(summary[c]["params"] for c in
                ("identity_storage", "shared_operator", "late_readout", "global_structural_prior"))
    assert parts == 99613
    # identity storage is the recomputed value, not a copied 36.6%
    assert summary["identity_storage"]["params"] == 36676
    assert summary["late_readout"]["params"] == 21633


# ---------------------------------------------------------------------------
# 3 / 14  instrumentation integrity
# ---------------------------------------------------------------------------

def test_instrumentation_does_not_change_prediction():
    _, _, _, etrain, _, _ = _encoded()
    graphs = etrain[:8]
    model = _model(0)
    from torch_geometric.loader import DataLoader

    loader = DataLoader(list(graphs), batch_size=8, shuffle=False)
    direct = []
    model.eval()
    with torch.no_grad():
        for b in loader:
            direct.append(model(b).detach().cpu().numpy().reshape(-1))
    direct = np.concatenate(direct)
    cap = pa.capture_intermediates(model, graphs, batch_size=8)
    assert np.abs(cap["yhat"] - direct).max() == 0.0


# ---------------------------------------------------------------------------
# 4-5  frequency uses official-train inputs only / never reads target
# ---------------------------------------------------------------------------

def test_token_frequency_counts_match_official_train_inputs():
    _, _, _, etrain, evalid, _ = _encoded()
    model = _model(0)
    freq = pa.stage_frequency(model, etrain, evalid)
    train_tok = pa._typed_tokens(etrain)
    expected = np.bincount(train_tok, minlength=freq["vocab_size"])
    rows = {r["token"]: r for r in freq["rows"]}
    assert len(rows) == freq["vocab_size"]
    for token in np.unique(train_tok):
        assert rows[int(token)]["train_occurrence"] == int(expected[token])
    assert freq["total_train_occurrences"] == int(train_tok.size)


def test_frequency_and_spectrum_do_not_reference_target():
    import inspect

    src = inspect.getsource(pa.stage_frequency)
    assert ".y" not in src and "target" not in src
    src_spec = inspect.getsource(pa.stage_embedding_spectrum)
    assert ".y" not in src_spec and "target" not in src_spec


# ---------------------------------------------------------------------------
# 6  SVD rank mechanically from energy
# ---------------------------------------------------------------------------

def test_spectral_rank_cutoffs_are_mechanical():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((50, 16))
    s = np.linalg.svd(M, compute_uv=False)
    rep = pa._spectral_report(M, s, "toy")
    s2 = s * s
    cum = np.cumsum(s2) / s2.sum()
    assert rep["rank_99pct"] == int(np.searchsorted(cum, 0.99) + 1)
    assert rep["rank_999pct"] == int(np.searchsorted(cum, 0.999) + 1)
    assert rep["rank_95pct"] <= rep["rank_99pct"] <= rep["rank_999pct"] <= 16


def test_build_lowrank_tables_uses_energy_ranks_and_full_is_identity():
    model = _model(0)
    tables, ranks = pa.build_lowrank_tables(model)
    # full effective table reproduces the checkpoint embedding exactly
    full_W = pa._effective_table(model)
    assert np.abs(tables["full"] - full_W).max() < 1e-5
    # r999 (rank 16 here) is the identity reconstruction
    assert ranks["r999"] == 16
    assert np.abs(tables["r999"] - tables["full"]).max() < 1e-4


# ---------------------------------------------------------------------------
# 7  low-rank perturbation leaves other parameters untouched
# ---------------------------------------------------------------------------

def test_lowrank_perturbation_does_not_mutate_nonlookup_params():
    _, _, _, etrain, _, _ = _encoded()
    graphs = etrain[:4]
    model = _model(0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()
              if not n.startswith("typed_embedding")}
    tables, _ = pa.build_lowrank_tables(model)
    pa.perturb_capture(model, graphs, tables, batch_size=4)
    for n, p in model.named_parameters():
        if n.startswith("typed_embedding"):
            continue
        assert torch.equal(p.detach(), before[n])
    # typed_embedding object restored
    assert hasattr(model.typed_embedding, "full")


# ---------------------------------------------------------------------------
# 8-9  checkpoint fingerprints / R dim
# ---------------------------------------------------------------------------

def test_checkpoint_fingerprints_match_files():
    inv = pa.checkpoint_inventory()
    for e in inv["entries"]:
        digest = hashlib.sha256(Path(e["state_path"]).read_bytes()).hexdigest()
        assert digest == e["sha256"]
        assert e["parameters"] == 99613
    assert inv["official_test_loaded"] is False


def test_R_is_302D():
    model = _model(0)
    assert int(model.head[0].in_features) == pa.R_DIM == 302
    _, _, _, etrain, _, _ = _encoded()
    cap = pa.capture_intermediates(model, etrain[:4], batch_size=4)
    assert cap["R"].shape[1] == 302
    # sub-blocks partition exactly
    assert pa.R_UNARY[1] + (pa.R_PAIR[1] - pa.R_PAIR[0]) + \
        (pa.R_GLOBAL[1] - pa.R_GLOBAL[0]) + (pa.R_TOPOLOGY[1] - pa.R_TOPOLOGY[0]) == 302


# ---------------------------------------------------------------------------
# 10-12  fixed small head / split discipline
# ---------------------------------------------------------------------------

def test_small_head_is_fixed_4135_no_search():
    from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import GenericReader

    head = GenericReader(pa.R_DIM, pa.SMALL_HEAD_HIDDEN)
    assert sum(p.numel() for p in head.parameters()) == pa.HEAD_SMALL_PARAMS == 4135
    # architecture is 302 -> 13 -> 13 -> 1
    dims = [m for m in head.net if isinstance(m, torch.nn.Linear)]
    assert [d.in_features for d in dims] == [302, 13, 13]
    assert [d.out_features for d in dims] == [13, 13, 1]


def test_head_split_is_official_train_internal_only():
    roles = pa.load_split_roles()
    assert int(np.sum(roles == "adapter_fit")) == 7200
    assert int(np.sum(roles == "adapter_selection")) == 800
    assert int(np.sum(roles == "train_probe")) == 2000
    manifest = pa.load_split_manifest()
    assert "official train only" in manifest["source"]


def test_valid_not_used_for_small_head_selection(monkeypatch):
    """The small head is selected on the 800 official-train selection split."""
    import tracks.ksvd.experiments.luyin16.zinc_graph_head_refit_capacity_decomposition as cd

    seen = {}
    original = cd.train_refit

    def spy(model, x_fit, y_fit, x_sel, y_sel, **kwargs):
        seen["n_fit"] = int(x_fit.shape[0])
        seen["n_sel"] = int(x_sel.shape[0])
        return original(model, x_fit, y_fit, x_sel, y_sel, **kwargs)

    monkeypatch.setattr(cd, "train_refit", spy)
    ext = np.load(pa.TRACK_ROOT / "results/optimized_manifold_broad_state_screen/state_exports/"
                  "optimized_frozen_state_export_v1_train_seed0.npz", allow_pickle=False)
    exv = np.load(pa.TRACK_ROOT / "results/optimized_manifold_broad_state_screen/state_exports/"
                  "optimized_frozen_state_export_v1_valid_seed0.npz", allow_pickle=False)
    pa.stage_head_screen(0, ext["R"].astype(np.float64), ext["target"].astype(np.float64),
                         exv["R"].astype(np.float64), exv["target"].astype(np.float64),
                         0.14642022556537995, None)
    assert seen["n_fit"] == 7200
    assert seen["n_sel"] == 800


# ---------------------------------------------------------------------------
# 13  official test is never loaded
# ---------------------------------------------------------------------------

def test_official_test_never_loaded():
    import inspect

    src = inspect.getsource(pa)
    assert '"test"' not in src
    assert "test.pickle" not in src
    assert pa.audit_protocol_lock()["official_test"] == "never loaded"
    assert pa.checkpoint_inventory()["official_test_loaded"] is False


# ---------------------------------------------------------------------------
# 15  gradient proxy never updates parameters
# ---------------------------------------------------------------------------

def test_gradient_audit_does_not_update_parameters():
    _, _, _, etrain, _, _ = _encoded()
    probe = pa.probe_graphs(etrain)[:64]
    model = _model(0)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    pa.stage_gradient(model, probe, batch_size=64)
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), before[n])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
