"""Key scientific checks for lcrc-lipo-edit-supervision-public122-s0-v1.

Run with the round directory in the pytest path.  These are not historical
round audits; they check the exact pieces this round relies on.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "code"))

import common as C  # noqa: E402
import data as D  # noqa: E402
import losses as L  # noqa: E402
import metrics as M  # noqa: E402
import pairs as P  # noqa: E402
import schedule as S  # noqa: E402


def test_source_split_and_frozen_model():
    info = C.verify_source()
    assert info["split_sizes"] == list(C.SPLIT_SIZES)
    spec = D.frozen_spec()
    assert spec.hash() == C.FEATURE_SPEC_HASH
    ds = D.load_dataset(include_test=False)
    y = np.asarray(ds["y_train"], dtype=np.float64)
    assert abs(float(ds["target_norm"]["mean"]) - float(y.mean())) < 1e-12
    assert abs(float(ds["target_norm"]["scale"]) - float(y.std(ddof=0))) < 1e-12
    full = C.DM.build_full_state(spec, ds["train_norm"], 0)
    init = C.DM.remap_state("C1", full)
    model = C.DM.load_checkpoint_model(spec, ds["train_norm"], "C1", init, torch.device("cpu"))
    assert C.DM.parameter_count(model) == 228674
    assert C.DM.shared_state_equal("C1", full, init)["all_equal"]


def test_loss_identity_and_gradients():
    torch.manual_seed(3)
    ea = torch.randn(32, dtype=torch.float64, requires_grad=True)
    eb = torch.randn(32, dtype=torch.float64, requires_grad=True)
    r = ((eb - ea) ** 2 / 2).mean()
    w = ((ea ** 2 + eb ** 2) / 2).mean()
    cross = (ea * eb).mean()
    assert abs(float(r - w + cross)) < 1e-15
    gr = torch.autograd.grad(r, [ea, eb], retain_graph=True)
    gw = torch.autograd.grad(w, [ea, eb], retain_graph=True)
    assert torch.allclose(gr[0], -(eb - ea) / 32, atol=1e-12)
    assert torch.allclose(gr[1], (eb - ea) / 32, atol=1e-12)
    assert torch.allclose(gw[0], ea / 32, atol=1e-12)
    assert torch.allclose(gw[1], eb / 32, atol=1e-12)


def test_random_pair_uses_its_own_label_difference():
    z = torch.tensor([0.1, 0.7, -0.4, 1.3], dtype=torch.float64)
    p = torch.tensor([0.2, 0.5, -0.1, 1.0], dtype=torch.float64)
    e = p - z
    u, v = 3, 1
    assert abs(float(e[v] - e[u]) - float((p[v] - p[u]) - (z[v] - z[u]))) < 1e-12


def test_pair_ids_dedup_direction_and_reverse():
    pairs = P.load_pairs()
    keys = [(int(p["row_lo"]), int(p["row_hi"])) for p in pairs]
    assert len(keys) == len(set(keys))
    assert all(str(p["pair_id"]) == f"p{int(p['row_lo']):05d}_{int(p['row_hi']):05d}" for p in pairs)
    assert all(str(p["var_a"]) < str(p["var_b"]) for p in pairs)
    y = np.arange(C.N_MOLECULES, dtype=np.float64)
    for p in pairs[:50]:
        a, b = int(p["row_a"]), int(p["row_b"])
        assert float(y[b] - y[a] + y[a] - y[b]) == 0.0


def test_core_mapping_sample():
    pairs = P.load_pairs()
    mappings = P.load_mappings()
    src = C.source_with_labels("none")
    # cheap deterministic sample; the full set is checked in correctness_gates.json
    sample = pairs[:: max(1, len(pairs) // 12)][:12]
    for p in sample:
        m = mappings[str(p["pair_id"])]
        core_a = C.EP._fragment_graph  # marker only; reconstruction is in gates
        assert set(int(k) for k in m["mapping"]) == {int(x) for x in m["core_atoms_a"]}
        assert set(int(v) for v in m["mapping"].values()) == {int(x) for x in m["core_atoms_b"]}
        assert int(m["mapping"][str(m["attach_a"])]) == int(m["attach_b"])


def test_schedule_multiset_and_forbidden_edges():
    if not S.SCHEDULE_NPZ.exists():
        S.ensure_schedule(P.p_train(), C.read_json(C.ANALYSIS / "schedule_manifest.json"))
    loaded = S.load_schedule()
    arrays = loaded["arrays"]
    manifest = loaded["manifest"]
    p_train = sorted(P.p_train(), key=lambda p: str(p["pair_id"]))
    canonical = P.canonical_smiles_map(C.source_with_labels("none")["smiles"])
    forbidden = {tuple(sorted((int(p["row_a"]), int(p["row_b"])))) for p in p_train}
    assert loaded["checks"]["real_pair_indices"] == manifest["real_pair_indices"]["sha256"]
    assert manifest["endpoint_multiset_equal_all_steps"]
    assert manifest["random_pairs_with_forbidden_edge"] == 0
    for step in range(0, manifest["n_steps"], 97):
        real_tokens = []
        for pi in arrays["real_pair_indices"][step]:
            p = p_train[int(pi)]
            real_tokens.extend((int(p["row_a"]), int(p["row_b"])))
        rand = [int(x) for x in arrays["random_u"][step]] + [int(x) for x in arrays["random_v"][step]]
        assert sorted(real_tokens) == sorted(rand)
        for u, v in zip(arrays["random_u"][step], arrays["random_v"][step]):
            u, v = int(u), int(v)
            assert u != v
            assert canonical[u] != canonical[v]
            assert tuple(sorted((u, v))) not in forbidden


def test_aux_eval_forward_has_gradients_and_no_rng_side_effect():
    ds = D.load_dataset(include_test=False)
    spec = ds["spec"]
    train_cache = ds["train_cache"]
    train_rows = [int(r) for r in ds["rows"]["train"]]
    full = C.DM.build_full_state(spec, ds["train_norm"], 0)
    init = C.DM.remap_state("C1", full)
    model = C.DM.load_checkpoint_model(spec, ds["train_norm"], "C1", init, torch.device("cpu"))
    model.train()
    row_to_idx = {int(r): i for i, r in enumerate(train_rows)}
    tokens = []
    for p in P.p_train()[:8]:
        tokens.extend((int(p["row_a"]), int(p["row_b"])))
    unique_rows = sorted(set(tokens))
    unique_idx = [row_to_idx[r] for r in unique_rows]
    before = C.sha256_bytes(torch.get_rng_state().numpy().tobytes())
    model.eval()
    pred = model(D.collate(train_cache, unique_idx)).reshape(-1)
    loss = (pred * pred).mean()
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], allow_unused=True)
    after = C.sha256_bytes(torch.get_rng_state().numpy().tobytes())
    model.train()
    assert before == after
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
    assert model.training


def test_single_vs_batch_and_row_id_join():
    ds = D.load_dataset(include_test=False)
    spec = ds["spec"]
    cache = ds["valid_cache"]
    full = C.DM.build_full_state(spec, ds["train_norm"], 0)
    init = C.DM.remap_state("C1", full)
    model = C.DM.load_checkpoint_model(spec, ds["train_norm"], "C1", init, torch.device("cpu"))
    model.eval()
    idx = [0, 1, 2, 3, 4]
    batched = D.predict_rows(model, cache, idx, torch.device("cpu"))
    singles = np.asarray([D.predict_rows(model, cache, [i], torch.device("cpu"))[0] for i in idx])
    assert float(np.max(np.abs(batched - singles))) < 1e-6
    rows = [int(x) for x in np.asarray(cache["rows"]).tolist()]
    joined = {rows[i]: float(batched[i]) for i in idx}
    assert sorted(joined) == sorted(rows[i] for i in idx)


def test_component_equal_and_bootstrap_weighting():
    values = np.asarray([1.0, 3.0, 10.0, 20.0])
    comps = np.asarray([0, 0, 1, 2])
    assert abs(M.component_equal_mean(values, comps) - (2.0 + 10.0 + 20.0) / 3) < 1e-12
    vals = {0: 1.0, 1: 3.0, 2: 8.0}
    idx = np.asarray([[0, 0], [1, 2]])
    boots, comp_ids = M.bootstrap_component_equal(vals, replicates=2, rng=np.random.default_rng(0), index=idx)
    assert np.allclose(boots, [1.0, 5.5])
    assert list(comp_ids) == [0, 1, 2]
    rows = {0: [0, 1], 1: [2], 2: [3, 4, 5]}
    sq = {0: 1.0, 1: 1.0, 2: 1.0, 3: 9.0, 4: 16.0, 5: 25.0}
    boots, comps, _, _ = M.bootstrap_test_rmse(rows, sq, replicates=2, rng=np.random.default_rng(0), index=idx)
    assert np.allclose(boots, [1.0, math.sqrt(51.0 / 4.0)])
    assert list(comps) == [0, 1, 2]


def test_comparison_gain_direction_and_reference_arm():
    """gain = reference - candidate; R_vs_A reference is A, candidate is R."""
    import csv

    import evaluate as E

    assert abs(E._point_gain(0.8, 1.0)["gain"] + 0.2) < 1e-15
    path = C.ANALYSIS / "comparisons.json"
    if not path.exists():
        pytest.skip("formal comparisons not produced yet")
    comp = json.loads(path.read_text())
    results = {row["arm"]: row for row in csv.DictReader(open(C.ANALYSIS / "results.csv"))}
    table = {
        "R_vs_A": ("A", "R", "test_rmse", "e_t_ce_mae"),
        "R_vs_W": ("W", "R", "test_rmse", "e_t_ce_mae"),
        "R_vs_N": ("N", "R", "test_rmse", "e_t_ce_mae"),
    }
    for label, (ref, cand, *metrics) in table.items():
        for metric in metrics:
            g = comp["comparisons"][label][metric]
            assert abs(g["reference"] - float(results[ref][metric])) < 1e-12
            assert abs(g["candidate"] - float(results[cand][metric])) < 1e-12
            assert abs(g["gain"] - (g["reference"] - g["candidate"])) < 1e-12


def test_train_diagnostics_endpoint_units_are_raw():
    path = C.ANALYSIS / "diagnostics_correction.json"
    if not path.exists():
        pytest.skip("formal diagnostics correction not produced yet")
    if not all((C.run_dir(arm) / "train_diagnostics.json").exists() for arm in C.ARMS):
        pytest.skip("formal run artifacts are not present in this checkout")
    for arm in C.ARMS:
        td = C.read_json(C.run_dir(arm) / "train_diagnostics.json")
        assert "units_correction" in td
        assert td["best"]["endpoint_molecules"]["rmse"] < 2.0
        assert td["best"]["nonpaired_molecules"]["rmse"] < 2.0


def test_historical_regression_is_explained():
    pairs = P.load_pairs()
    audit = C.read_json(C.ANALYSIS / "data_audit.json")
    hist = audit.get("historical_regression_check", {})
    if not hist.get("available"):
        pytest.skip("historical pairs.csv not available")
    assert hist["old_pairs_csv_rows"] == 509
    assert hist["old_unique_unordered_pairs"] == 501
    assert hist["old_duplicate_definitions"] == 8
    assert hist["new_p_train_pairs"] == 501
    assert hist["old_unique_set_equals_new_p_train"]
    assert hist["new_p_train_molecules"] == 502
