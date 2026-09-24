"""Focused CPU tests for E2E-DictEnv-T1 (no ZINC, no GPU)."""

from __future__ import annotations

import json

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as e2e_v0
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_t1 as t1
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_t1 as runner


def _synthetic_data(n: int, edges: list[tuple[int, int]], *, seed: int = 0):
    graph = from_edges(n, edges)
    edge_types = {graph.edge_key(a, b): (index % t1.v0.BOND_CATEGORIES) for index, (a, b) in enumerate(edges)}
    incidence = e2e_v0.env_incidence(graph, edge_types)
    rng = np.random.default_rng(seed)
    data = Data()
    data.num_nodes = n
    data.dict_phi = torch.as_tensor(rng.standard_normal((n, t1.PHI_DIM)).astype(np.float32))
    data.dict_atom = torch.as_tensor(rng.integers(0, t1.ATOM_CATEGORIES, size=n), dtype=torch.long)
    data.patch_cont = torch.as_tensor(rng.standard_normal((n, t1.COARSE_DIM)).astype(np.float32))
    data.env_occ_node = incidence["occ_node"]
    data.env_occ_root = incidence["occ_root"]
    data.env_occ_shell = incidence["occ_shell"]
    data.env_bond_root = incidence["bond_root"]
    data.env_bond_shellpair = incidence["bond_shellpair"]
    data.env_bond_type = incidence["bond_type"]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if pairs:
        data.pair_index = torch.as_tensor(np.asarray(pairs, dtype=np.int64).T)
        data.pair_relation = torch.randn(len(pairs), t1.RELATION_WIDTH)
        data.pair_bucket = torch.as_tensor(rng.integers(0, t1.DISTANCE_BUCKETS, size=len(pairs)), dtype=torch.long)
    else:
        data.pair_index = torch.empty((2, 0), dtype=torch.long)
        data.pair_relation = torch.empty((0, t1.RELATION_WIDTH))
        data.pair_bucket = torch.empty((0,), dtype=torch.long)
    data.global_context = torch.randn(1, t1.GLOBAL_WIDTH)
    data.topology_features = torch.randn(1, 25)
    data.y = torch.tensor([0.5])
    return data


def _synthetic_batch(n_molecules: int = 3):
    molecules = [_synthetic_data(4, [(0, 1), (1, 2), (2, 3)], seed=index) for index in range(n_molecules)]
    return e2e_v0.env_collate(molecules)


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def test_parameter_accounting_matches_preregistration():
    a1 = t1.total_parameter_count(t1.CANDIDATES["a1"])
    a2 = t1.total_parameter_count(t1.CANDIDATES["a2"])
    a3 = t1.total_parameter_count(t1.CANDIDATES["a3"])
    assert t1.local_parameter_count(t1.CANDIDATES["a1"]) == {"dictionary": 2080, "binding": 4032, "decoder": 44596, "subtotal": 50708}
    assert t1.local_parameter_count(t1.CANDIDATES["a2"]) == {"dictionary": 2080, "binding": 2880, "decoder": 45813, "subtotal": 50773}
    assert t1.local_parameter_count(t1.CANDIDATES["a3"]) == {"dictionary": 2080, "binding": 3840, "decoder": 44940, "subtotal": 50860}
    assert (a1["whole_model"], a1["difference_vs_fec_s1"]) == (66067, -103)
    assert (a2["whole_model"], a2["difference_vs_fec_s1"]) == (66132, -38)
    assert (a3["whole_model"], a3["difference_vs_fec_s1"]) == (66219, 49)
    for key in t1.CANDIDATE_ORDER:
        assert t1.total_parameter_count(t1.CANDIDATES[key])["backend_total"] == 15359


def test_model_parameter_counts_and_init_matching():
    for key in t1.CANDIDATE_ORDER:
        candidate = t1.CANDIDATES[key]
        sparse = t1.build_model(t1.SPARSE_ARM, candidate, seed=0)
        dense = t1.build_model(t1.DENSE_ARM, candidate, seed=0, reference_state=sparse.state_dict())
        expected = t1.total_parameter_count(candidate)["whole_model"]
        assert sum(p.numel() for p in sparse.parameters()) == expected
        assert sum(p.numel() for p in dense.parameters()) == expected
        for name, value in sparse.state_dict().items():
            assert torch.equal(value, dense.state_dict()[name]), (key, name)


def test_dictionary_init_is_the_sdb_artifact():
    from tracks.ksvd.experiments.luyin16 import zinc_sdb_v0 as zsdb

    D_sdb, _rand, _pca = zsdb.load_dictionary()
    model = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES["a2"], seed=0)
    assert np.array_equal(model.D.detach().numpy(), np.asarray(D_sdb, dtype=np.float32))


# ---------------------------------------------------------------------------
# coding / forward
# ---------------------------------------------------------------------------


def test_coding_semantics_and_sparsity():
    for key in t1.CANDIDATE_ORDER:
        sparse = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES[key], seed=0)
        dense = t1.build_model(t1.DENSE_ARM, t1.CANDIDATES[key], seed=0)
        phi = torch.randn(16, t1.PHI_DIM)
        alpha = sparse.code(phi)
        assert int((alpha.abs() > 0).sum(dim=1).max()) <= t1.SPARSITY
        dbar = e2e_v0.normalized_dictionary(sparse.D)
        assert torch.allclose(dense.code(phi), phi @ dbar, atol=1e-6)
        assert torch.allclose(sparse.reconstruct(phi, alpha), alpha @ dbar.t(), atol=1e-6)


def test_forward_shapes_and_coarse_usage():
    batch = _synthetic_batch()
    for key in t1.CANDIDATE_ORDER:
        model = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES[key], seed=0).eval()
        with torch.no_grad():
            prediction, aux = model(batch, return_aux=True)
        assert prediction.shape[0] == 3
        assert aux["E"].shape == (int(batch.num_nodes), t1.ENV_DIM)
        assert aux["coord"].shape == (int(batch.num_nodes), t1.K_ATOMS)


def test_environment_freeze_under_pair_relation_mutation():
    batch = _synthetic_batch(2)
    model = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES["a3"], seed=0).eval()
    captured = {}
    handle = model.env_mlp.register_forward_hook(lambda _m, _i, out: captured.setdefault("E", out.detach().clone()))
    try:
        with torch.no_grad():
            model(batch)
        before = captured["E"].clone()
        mutated = batch.clone()
        mutated.pair_relation = torch.randn_like(mutated.pair_relation)
        with torch.no_grad():
            model(mutated)
        after = captured["E"].clone()
    finally:
        handle.remove()
    assert torch.equal(before, after)


def test_zero_code_independent_of_phi_but_coarse_still_used():
    batch = _synthetic_batch(2)
    model = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES["a1"], seed=0).eval()
    with torch.no_grad():
        _p, aux_a = model(batch, coord_zero=True, return_aux=True)
        mutated = batch.clone()
        mutated.dict_phi = torch.randn_like(mutated.dict_phi)
        _p2, aux_b = model(mutated, coord_zero=True, return_aux=True)
        _p3, aux_c = model(mutated, coord_zero=False, return_aux=True)
    assert torch.equal(aux_a["E"], aux_b["E"])
    assert not torch.equal(aux_b["E"], aux_c["E"])


def test_pair_encoder_called_once():
    batch = _synthetic_batch(2)
    model = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES["a2"], seed=0).eval()
    counts = {"pair": 0, "relation": 0}

    def _hook(name):
        def _inner(_m, _i, _o):
            counts[name] += 1
        return _inner

    handles = [
        model.pair_encoder.register_forward_hook(_hook("pair")),
        model.relation_encoder.register_forward_hook(_hook("relation")),
    ]
    try:
        with torch.no_grad():
            model(batch)
    finally:
        for handle in handles:
            handle.remove()
    assert counts == {"pair": 1, "relation": 1}


def test_no_forbidden_modules_in_state_dict():
    for key in t1.CANDIDATE_ORDER:
        model = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES[key], seed=0)
        for name in model.state_dict():
            lowered = name.lower()
            assert "attention" not in lowered
            assert "lstm" not in lowered and "gru" not in lowered
            assert "typed" not in lowered and "parent_embedding" not in lowered
            assert "local_env_adapter" not in lowered
            assert "bond" not in lowered


def test_gradient_reaches_dictionary():
    batch = _synthetic_batch(2)
    for key in t1.CANDIDATE_ORDER:
        model = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES[key], seed=0).train()
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(batch), batch.y.view(-1))
        loss.backward()
        assert model.D.grad is not None
        assert float(model.D.grad.norm()) > 0.0


def test_slot_environment_uses_every_shell():
    batch = _synthetic_batch(2)
    model = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES["a2"], seed=0).eval()
    with torch.no_grad():
        _p, aux = model(batch, return_aux=True)
    # the slot candidate concatenates three per-shell blocks of width r_dict
    m = model.dictionary_environment(aux["coord"], batch, None)
    assert m.shape[1] == 3 * t1.CANDIDATES["a2"].r_dict
    # a pooled candidate produces one block of width r_dict
    pooled = t1.build_model(t1.SPARSE_ARM, t1.CANDIDATES["a1"], seed=0).eval()
    with torch.no_grad():
        _p2, aux2 = pooled(batch, return_aux=True)
    m2 = pooled.dictionary_environment(aux2["coord"], batch, None)
    assert m2.shape[1] == t1.CANDIDATES["a1"].r_dict


def test_env_collate_offsets_occurrences():
    molecules = [
        _synthetic_data(4, [(0, 1), (1, 2), (2, 3)], seed=0),
        _synthetic_data(3, [(0, 1), (1, 2)], seed=1),
    ]
    batch = e2e_v0.env_collate(molecules)
    first = int(molecules[0].num_nodes)
    occ_first = int(molecules[0].env_occ_node.shape[0])
    assert int(batch.env_occ_node[occ_first:].min()) >= first
    assert int(batch.env_occ_root.max()) == int(batch.num_nodes) - 1
    assert batch.patch_cont.shape[1] == t1.COARSE_DIM


# ---------------------------------------------------------------------------
# tuning orchestration (per-candidate parallel runs)
# ---------------------------------------------------------------------------


def _fake_stage_a_payload(candidate_id: str, soup: float, best: float, epoch: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "lambda_rec": 135.83,
        "soup": {"soup_valid_mae": soup, "soup_valid_rec": 1e-4, "members": [epoch - 4, epoch - 2, epoch, epoch + 2, epoch + 4]},
        "best_valid_mae": best,
        "best_epoch": epoch,
        "train_mae_at_best": 0.09,
        "wall_clock_s": 1000.0,
        "parameter_accounting": {"whole_model": 66132},
    }


def test_stage_a_rows_follow_candidate_order_and_select_min(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    for key, soup, best, epoch in [("a3", 0.130, 0.140, 100), ("a1", 0.124, 0.130, 90), ("a2", 0.126, 0.135, 95)]:
        (tmp_path / f"stage_a_{key}.json").write_text(json.dumps(_fake_stage_a_payload(key.upper(), soup, best, epoch)))
    rows = runner._stage_a_rows()
    assert [row["candidate_key"] for row in rows] == ["a1", "a2", "a3"]
    monkeypatch.setattr(runner, "_health_gate_ok", lambda *_a, **_k: True)
    selection = runner.resolve_stage_a()
    assert selection["winner"] == "a1"
    monkeypatch.setattr(runner, "_health_gate_ok", lambda key, *_a, **_k: key == "a3")
    assert runner.resolve_stage_a()["winner"] == "a3"


def test_stage_a_only_trains_one_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    calls: list[str] = []

    def _fake_train_run(arm, candidate_key, seed, lambda_rec, epochs, tag, device, *args, **kwargs):
        calls.append(tag)
        (tmp_path / f"{tag}.json").write_text(json.dumps(_fake_stage_a_payload(candidate_key.upper(), 0.12, 0.13, 100)))
        return json.loads((tmp_path / f"{tag}.json").read_text())

    monkeypatch.setattr(runner, "train_run", _fake_train_run)
    runner.stage_a_stage(device="cpu", only="a2")
    assert calls == ["stage_a_a2"]
    assert not (tmp_path / "stage_a_summary.json").exists()
    runner.stage_a_stage(device="cpu")
    assert calls == ["stage_a_a2", "stage_a_a1", "stage_a_a2", "stage_a_a3"]
    assert (tmp_path / "stage_a_summary.json").exists()


def test_stage_b_only_trains_one_arm_and_reads_per_tag_files(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    (tmp_path / "stage_a_a2.json").write_text(json.dumps(_fake_stage_a_payload("A2_COARSE146_SLOT48", 0.126, 0.131, 237)))
    monkeypatch.setattr(runner, "_health_gate_ok", lambda *_a, **_k: True)
    calls: list[str] = []

    def _fake_train_run(arm, candidate_key, seed, lambda_rec, epochs, tag, device, *args, **kwargs):
        calls.append(tag)
        payload = _fake_stage_a_payload(candidate_key.upper(), 0.125, 0.132, 234)
        payload["soup"]["soup_valid_rec"] = 1.4e-4
        (tmp_path / f"{tag}.json").write_text(json.dumps(payload))
        return payload

    monkeypatch.setattr(runner, "train_run", _fake_train_run)
    runner.stage_b_stage(device="cpu", only="b2")
    assert calls == ["stage_b_lambda025"]
    resolved = runner.resolve_stage_b()
    assert resolved["winner"] == "a2"
    assert [row["tag"] for row in resolved["rows"]] == ["stage_a_a2", "stage_b_lambda025"]
    assert resolved["chosen"]["tag"] == "stage_b_lambda025"
