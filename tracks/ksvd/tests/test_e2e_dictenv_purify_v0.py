"""Focused CPU tests for E2E-DictEnv-Purify-v0 (no ZINC, no GPU).

Covers the frozen pre-registration test list: semantic-refactor equivalence,
constant-channel semantics, sparsity, dictionary definition, invariance,
anchor purity, global/pair/reader freeze, matched init, parameter ledger,
official-test blocker and GPU explicitness.
"""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_purify_v0 as pur
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_purify_v0 as runner


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------


def _random_dictionary(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    D = rng.randn(pur.PHI_DIM, pur.K_ATOMS).astype(np.float32)
    return D / np.maximum(np.linalg.norm(D, axis=0, keepdims=True), 1e-6)


def _synthetic_data(
    n: int,
    edges: list[tuple[int, int]],
    *,
    seed: int = 0,
    atom: torch.Tensor | None = None,
    bond_types: dict | None = None,
) -> Data:
    graph = from_edges(n, edges)
    edge_types = (
        {graph.edge_key(a, b): (index % pur.BOND_CATEGORIES) for index, (a, b) in enumerate(edges)}
        if bond_types is None
        else bond_types
    )
    incidence = v0.env_incidence(graph, edge_types)
    rng = np.random.default_rng(seed)
    data = Data()
    data.num_nodes = n
    data.dict_phi = torch.as_tensor(rng.standard_normal((n, pur.PHI_DIM)).astype(np.float32))
    if atom is None:
        data.dict_atom = torch.as_tensor(rng.integers(0, pur.ATOM_CATEGORIES, size=n), dtype=torch.long)
    else:
        data.dict_atom = atom.long()
    for key in ("occ_node", "occ_root", "occ_shell", "bond_root", "bond_shellpair", "bond_type", "bond_u", "bond_v"):
        setattr(data, f"env_{key}", incidence[key])
    data.anchor = p1.build_anchor_raw(
        data.dict_atom,
        data.env_occ_node,
        data.env_occ_root,
        data.env_bond_root,
        data.env_bond_type,
        n,
    )
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    data.pair_index = torch.as_tensor(np.asarray(pairs, dtype=np.int64).T)
    data.pair_relation = torch.randn(len(pairs), p1.RELATION_WIDTH_RAW)
    data.pair_bucket = torch.as_tensor(rng.integers(0, pur.DISTANCE_BUCKETS, size=len(pairs)), dtype=torch.long)
    data.global_context = torch.randn(1, pur.GLOBAL_WIDTH)
    data.topology_features = torch.randn(1, pur.TOPOLOGY_HIDDEN + 9)
    data.y = torch.tensor([0.5])
    return data


def _synthetic_batch(n_molecules: int = 3, n_nodes: int = 4):
    molecules = [
        _synthetic_data(n_nodes, [(i, i + 1) for i in range(n_nodes - 1)], seed=index)
        for index in range(n_molecules)
    ]
    return p1.env_collate(molecules)


def _models(seed: int = 0) -> tuple[pur.PurifyV0Model, pur.PurifyV0Model]:
    D = _random_dictionary(seed=seed)
    reference = pur.build_model(pur.reference_config(), D, seed=seed).eval()
    purified = pur.build_model(pur.purified_config(), D, seed=seed).eval()
    return reference, purified


# ---------------------------------------------------------------------------
# stage 0 — semantic refactor equivalence
# ---------------------------------------------------------------------------


def test_stage0_semantic_refactor_matches_p2_model_bit_for_bit():
    D = _random_dictionary()
    legacy = p2.build_model(p2.P2Config("H1", "h1", 48, 32, 8, 0.25, 320, "sdb32"), D, seed=0).eval()
    refactored = pur.build_model(pur.reference_config(), D, seed=0).eval()
    pur.load_p2_state(refactored, legacy.state_dict())
    batch = _synthetic_batch(2)
    with torch.no_grad():
        legacy_prediction = legacy(batch)
        prediction, aux = refactored(batch, return_aux=True)
    assert torch.equal(legacy_prediction, prediction)
    assert aux["node_slots"].shape == (batch.dict_phi.shape[0], pur.N_SHELLS, pur.D_A)
    assert aux["edge_slots"].shape == (batch.dict_phi.shape[0], pur.SHELLPAIR_CLASSES, pur.D_E)
    assert aux["unified"].shape == (2, pur.UNIFIED_DIM)


def test_stage0_equivalence_runner_stage(monkeypatch):
    """Real-data equivalence when the frozen caches/artifacts exist, else skip."""
    cache = runner.env_cache_path("valid")
    if not cache.exists():
        pytest.skip("frozen validity cache not present locally")
    try:
        payload = runner.semantic_refactor_equivalence(n_molecules=4, device="cpu")
    except FileNotFoundError as error:  # encoded ZINC cache missing
        pytest.skip(f"encoded cache unavailable: {error}")
    assert payload["passed"], payload["comparisons_max_abs"]
    assert payload["max_abs"] <= 1.0e-6
    assert payload["comparisons_max_abs"]["prediction"] <= 1.0e-6


# ---------------------------------------------------------------------------
# constant channel semantics
# ---------------------------------------------------------------------------


def test_constant_channel_is_not_part_of_alpha():
    reference, purified = _models()
    batch = _synthetic_batch(2)
    with torch.no_grad():
        alpha = purified.code(batch.dict_phi)
        aux_reference = reference(batch, return_aux=True)[1]
    assert alpha.shape[1] == pur.K_ATOMS
    assert purified.measure.W_A_S.shape[0] == pur.K_ATOMS + 1
    assert purified.measure.W_E_S.shape[0] == 3 * pur.K_ATOMS + 1
    # the constant node column is exactly the row-0 contribution of W_A_S
    q = torch.nn.functional.one_hot(batch.dict_atom, num_classes=pur.ATOM_CATEGORIES).float()
    with torch.no_grad():
        units_on = purified.measure.node_units(alpha, batch, constant_scale=1.0)
        units_off = purified.measure.node_units(alpha, batch, constant_scale=0.0)
    attribute = q[batch.env_occ_node]
    expected = (purified.measure.W_A_S[0] * (attribute @ purified.measure.W_A_C)) / math.sqrt(pur.D_A)
    assert torch.allclose(units_on - units_off, expected, atol=1e-6)
    # the reference has no constant channel at all
    assert aux_reference["node_slots"].shape[-1] == pur.D_A
    assert reference.config.node_structure_dim == pur.K_ATOMS


def test_constant_channel_not_counted_toward_sparsity():
    _reference, purified = _models()
    batch = _synthetic_batch(2)
    with torch.no_grad():
        alpha = purified.code(batch.dict_phi)
        l0 = (alpha.abs() > 0).sum(dim=1)
    assert int(l0.max()) <= pur.SPARSITY
    assert alpha.shape[1] == pur.K_ATOMS
    assert purified.config.node_structure_dim == pur.K_ATOMS + 1


def test_max_l0_within_sparsity_both_flavours():
    reference, purified = _models()
    batch = _synthetic_batch(3)
    for model in (reference, purified):
        with torch.no_grad():
            l0 = (model.code(batch.dict_phi).abs() > 0).sum(dim=1)
        assert int(l0.max()) <= pur.SPARSITY


def test_dictionary_reconstruction_definition_unchanged():
    D = _random_dictionary()
    legacy = p2.build_model(p2.P2Config("H1", "h1", 48, 32, 8, 0.25, 320, "sdb32"), D, seed=0).eval()
    refactored = pur.build_model(pur.reference_config(), D, seed=0).eval()
    pur.load_p2_state(refactored, legacy.state_dict())
    phi = torch.randn(16, pur.PHI_DIM)
    with torch.no_grad():
        coord = refactored.code(phi)
        expected = coord @ v0.normalized_dictionary(refactored.D).t()
        assert torch.equal(refactored.reconstruct(coord), expected)
        assert torch.equal(refactored.reconstruct(coord), legacy.reconstruct(phi, coord))
        assert torch.allclose(
            refactored.reconstruction_loss(phi, coord),
            legacy.reconstruction_loss(phi, coord),
        )
    assert pur.reference_config().K == pur.K_ATOMS
    assert pur.reference_config().s == pur.SPARSITY
    assert pur.reference_config().iht_steps == pur.IHT_STEPS


def test_chemistry_relabel_leaves_phi_and_alpha_unchanged():
    _reference, purified = _models()
    data = _synthetic_data(5, [(0, 1), (1, 2), (2, 3), (3, 4)], seed=3)
    relabelled = data.clone()
    rng = np.random.default_rng(7)
    atom_permutation = torch.as_tensor(rng.permutation(pur.ATOM_CATEGORIES), dtype=torch.long)
    bond_permutation = torch.as_tensor(rng.permutation(pur.BOND_CATEGORIES), dtype=torch.long)
    relabelled.dict_atom = atom_permutation[data.dict_atom % pur.ATOM_CATEGORIES]
    relabelled.env_bond_type = bond_permutation[data.env_bond_type % pur.BOND_CATEGORIES]
    batch_a = p1.env_collate([data])
    batch_b = p1.env_collate([relabelled])
    with torch.no_grad():
        alpha_a = purified.code(batch_a.dict_phi)
        alpha_b = purified.code(batch_b.dict_phi)
        env_a = purified(batch_a, return_aux=True)[1]["E"]
        env_b = purified(batch_b, return_aux=True)[1]["E"]
    assert torch.equal(batch_a.dict_phi, batch_b.dict_phi)
    assert torch.equal(alpha_a, alpha_b)
    assert batch_a.dict_phi.shape[1] == pur.PHI_DIM
    assert not torch.equal(env_a, env_b)


# ---------------------------------------------------------------------------
# invariance
# ---------------------------------------------------------------------------


def _relabel(data: Data, permutation: np.ndarray) -> Data:
    """Relabel node ids by ``permutation`` (new_id = permutation[old_id])."""
    out = data.clone()
    mapping = torch.as_tensor(permutation, dtype=torch.long)
    inverse = torch.empty_like(mapping)
    inverse[mapping] = torch.arange(mapping.numel())
    out.dict_phi = data.dict_phi[inverse]
    out.dict_atom = data.dict_atom[inverse]
    out.anchor = data.anchor[inverse]
    for key in ("env_occ_node", "env_bond_u", "env_bond_v"):
        setattr(out, key, mapping[getattr(data, key)])
    # roots use the local node index space as well
    out.env_occ_root = mapping[data.env_occ_root]
    out.env_bond_root = mapping[data.env_bond_root]
    out.pair_index = mapping[data.pair_index]
    return out


def test_graph_relabel_invariance():
    _reference, purified = _models()
    n = 6
    data = _synthetic_data(n, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3)], seed=11)
    batch_a = p1.env_collate([data])
    permutation = np.random.default_rng(5).permutation(n)
    batch_b = p1.env_collate([_relabel(data, permutation)])
    with torch.no_grad():
        prediction_a = purified(batch_a)
        prediction_b = purified(batch_b)
    assert torch.allclose(prediction_a, prediction_b, atol=1e-6)


def test_node_permutation_invariance():
    _reference, purified = _models()
    n = 5
    data = _synthetic_data(n, [(0, 1), (1, 2), (2, 3), (3, 4)], seed=13)
    batch = p1.env_collate([data])
    rolled = _relabel(data, np.roll(np.arange(n), 2))
    with torch.no_grad():
        prediction_a = purified(batch)
        prediction_b = purified(p1.env_collate([rolled]))
    assert torch.allclose(prediction_a, prediction_b, atol=1e-6)


def test_edge_endpoint_symmetry():
    _reference, purified = _models()
    batch = _synthetic_batch(2)
    with torch.no_grad():
        alpha = purified.code(batch.dict_phi)
        left = alpha[batch.env_bond_u]
        right = alpha[batch.env_bond_v]
        forward = purified.measure._edge_structure(left, right, 1.0)
        backward = purified.measure._edge_structure(right, left, 1.0)
    assert torch.equal(forward, backward)
    assert forward.shape[1] == 3 * pur.K_ATOMS + 1


# ---------------------------------------------------------------------------
# anchor purity
# ---------------------------------------------------------------------------


def test_local_anchor_chemistry_truly_absent():
    _reference, purified = _models()
    batch = _synthetic_batch(2)
    poisoned = batch.clone()
    poisoned.anchor = batch.anchor.clone()
    poisoned.anchor[:, : pur.ANCHOR_SIZE.start] = torch.randn_like(poisoned.anchor[:, : pur.ANCHOR_SIZE.start])
    poisoned.dict_atom = (batch.dict_atom + 3) % pur.ATOM_CATEGORIES
    with torch.no_grad():
        clean = purified(batch)
        poisoned_prediction = purified(poisoned)
    assert purified.config.anchor_dim == pur.ANCHOR_DIM_PURIFIED == 2
    assert torch.allclose(clean, poisoned_prediction, atol=1e-6), "anchor chemistry must not reach the purified model"
    # the reference model does read those 60 chemistry dims
    reference = pur.build_model(pur.reference_config(), _random_dictionary(), seed=0).eval()
    with torch.no_grad():
        assert not torch.allclose(reference(batch), reference(poisoned), atol=1e-6)


def test_only_size_remains_in_local_structural_scalar_slot():
    _reference, purified = _models()
    n = 5
    data = _synthetic_data(n, [(0, 1), (1, 2), (2, 3), (3, 4)], seed=17)
    anchor = purified.composer.local_anchor(data, purified.config.anchor_dim)
    node_count = torch.zeros(n).index_add_(0, data.env_occ_root, torch.ones_like(data.env_occ_root, dtype=torch.float))
    edge_count = torch.zeros(n).index_add_(0, data.env_bond_root, torch.ones_like(data.env_bond_root, dtype=torch.float))
    assert anchor.shape == (n, 2)
    assert torch.allclose(anchor[:, 0], torch.log1p(node_count), atol=1e-6)
    assert torch.allclose(anchor[:, 1], torch.log1p(edge_count), atol=1e-6)
    assert pur.ANCHOR_SIZE == slice(60, 62)


# ---------------------------------------------------------------------------
# frozen global / pair / reader
# ---------------------------------------------------------------------------


def test_global_chemistry_unchanged():
    reference, purified = _models()
    assert pur.GLOBAL_ZEROTH_ORDER_ATTRIBUTE == slice(30, 62)
    assert pur.GLOBAL_ZEROTH_ORDER_ATTRIBUTE.stop - pur.GLOBAL_ZEROTH_ORDER_ATTRIBUTE.start == 32
    for model in (reference, purified):
        assert model.composer.global_encoder.layers[0].in_features == pur.GLOBAL_WIDTH == 62
    batch = _synthetic_batch(2)
    mutated = batch.clone()
    mutated.global_context = batch.global_context.clone()
    mutated.global_context[:, pur.GLOBAL_ZEROTH_ORDER_ATTRIBUTE] += 3.0
    with torch.no_grad():
        assert not torch.equal(purified(batch), purified(mutated))


def test_global_topology_unchanged():
    reference, purified = _models()
    assert pur.GLOBAL_STRUCTURAL_INVARIANT == slice(0, 30)
    for model in (reference, purified):
        assert model.composer.topology_encoder[0].in_features == pur.TOPOLOGY_IN == 25
        assert model.composer.topology_encoder[2].out_features == pur.TOPOLOGY_OUT == 8
    batch = _synthetic_batch(2)
    mutated = batch.clone()
    mutated.topology_features = batch.topology_features + 5.0
    with torch.no_grad():
        assert not torch.equal(purified(batch), purified(mutated))


def test_pair_relation_unchanged():
    reference, purified = _models()
    assert list(pur.P1_RELATION_INDICES) == list(range(14)) + [18]
    assert pur.RELATION_WIDTH == 15
    for model in (reference, purified):
        assert model.composer.relation_encoder.layers[0].in_features == pur.RELATION_WIDTH
        assert model.composer.distance_gate.num_embeddings == pur.DISTANCE_BUCKETS == 5
        assert model.composer.pair_encoder.layers[0].in_features == 4 * pur.PAIR_HIDDEN
    batch = _synthetic_batch(2)
    mutated = batch.clone()
    mutated.pair_relation = batch.pair_relation.clone()
    mutated.pair_relation[:, 14:18] = torch.randn_like(mutated.pair_relation[:, 14:18])
    mutated.pair_relation[:, 19:23] = torch.randn_like(mutated.pair_relation[:, 19:23])
    with torch.no_grad():
        assert torch.equal(purified(batch), purified(mutated))


def test_reader_unchanged():
    reference, purified = _models()
    assert pur.UNIFIED_DIM == 302
    assert pur.READER_HIDDEN == (13, 13)
    for model in (reference, purified):
        layers = model.reader.reader.net
        assert layers[0].in_features == 302
        assert layers[0].out_features == 13
        assert layers[2].out_features == 13
        assert layers[4].out_features == 1
    assert sum(p.numel() for p in reference.reader.parameters()) == 4135


def test_module_source_has_no_forbidden_bypass():
    tree = ast.parse(Path(pur.__file__).read_text(encoding="utf-8"))
    forbidden = {"patch_cont", "atom_shell", "bond_shell"}
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)} | {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert not (names & forbidden)


# ---------------------------------------------------------------------------
# matched init / parameters
# ---------------------------------------------------------------------------


def test_shared_init_identity_and_constant_rows_deterministic():
    D = _random_dictionary()
    reference, candidate, report = pur.build_matched_pair(D, seed=0)
    assert report["max_abs_diff"] == 0.0
    assert report["all_shared_identical"] is True
    assert report["anchor_size_slice_identical"] is True
    assert report["n_shared_tensors"] == len(report["shared_keys"])
    # every same-shape shared tensor is exactly identical
    reference_state, candidate_state = reference.state_dict(), candidate.state_dict()
    for key in report["shared_keys"]:
        assert torch.equal(reference_state[key], candidate_state[key]), key
    # dictionary + reader are shared verbatim
    assert torch.equal(reference_state["basis.D"], candidate_state["basis.D"])
    assert torch.equal(reference_state["reader.reader.net.0.weight"], candidate_state["reader.reader.net.0.weight"])
    # candidate-only constant rows are deterministic across constructions
    _r2, candidate2, _report2 = pur.build_matched_pair(D, seed=0)
    assert torch.equal(candidate.measure.W_A_S[0], candidate2.measure.W_A_S[0])
    assert torch.equal(candidate.measure.W_E_S[0], candidate2.measure.W_E_S[0])
    assert torch.equal(candidate.measure.W_A_S[1:], reference.measure.W_A_S)
    assert torch.equal(candidate.measure.W_E_S[1:], reference.measure.W_E_S)


def test_parameter_count_ledger_and_rule():
    ledger = pur.architecture_ledger()
    assert ledger["reference"]["whole_model"] == 97487
    assert ledger["purified"]["whole_model"] == 95711
    assert ledger["delta"]["absolute"] == -1776
    assert ledger["delta"]["within_parameter_rule"] is True
    assert ledger["purified"]["whole_model"] <= ledger["reference"]["whole_model"]
    D = _random_dictionary()
    for flavor, config in (("reference", pur.reference_config()), ("purified", pur.purified_config())):
        model = pur.build_model(config, D, seed=0)
        actual = sum(parameter.numel() for parameter in model.parameters())
        assert actual == ledger[flavor]["whole_model"], flavor
    assert pur.parameter_ledger(pur.purified_config())["layers"] == {
        "StructuralBasis": 2080,
        "AttributedLocalMeasure": 10704,
        "StaticComposer": 78792,
        "PropertyReader": 4135,
    }


def test_purity_audit_counts():
    reference = pur.purity_audit(pur.reference_config())
    purified = pur.purity_audit(pur.purified_config())
    assert reference["local_chemistry_entry_points"] == 5
    assert reference["local_raw_chemistry_bypass_count"] == 3
    assert purified["local_chemistry_entry_points"] == 2
    assert purified["local_raw_chemistry_bypass_count"] == 0
    assert reference["global_chemistry_entry_points"] == purified["global_chemistry_entry_points"] == 1
    assert purified["constant_structural_channel"]["present"] is True
    assert purified["constant_structural_channel"]["counted_toward_top_s"] is False
    assert reference["constant_structural_channel"]["present"] is False


# ---------------------------------------------------------------------------
# gradients / device / split discipline
# ---------------------------------------------------------------------------


def test_purified_gradients_reach_every_core_mechanism():
    _reference, purified = _models()
    batch = _synthetic_batch(3)
    purified.train()
    purified.zero_grad(set_to_none=True)
    prediction, aux = purified(batch, return_aux=True)
    loss = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1)) + purified.reconstruction_loss(
        aux["phi"], aux["coord"]
    )
    loss.backward()
    assert float(purified.basis.D.grad.norm()) > 0.0
    assert float(purified.measure.W_A_C.grad.norm()) > 0.0
    assert float(purified.measure.W_E_C.grad.norm()) > 0.0
    assert float(purified.measure.W_A_S.grad[0].norm()) > 0.0
    assert float(purified.measure.W_E_S.grad[0].norm()) > 0.0
    assert float(purified.measure.W_A_S.grad[1:].norm()) > 0.0


def test_official_test_blocker():
    assert runner.OFFICIAL_TEST_BLOCKED is True
    with pytest.raises(PermissionError):
        runner.load_split("test")
    with pytest.raises(PermissionError):
        runner.load_split("official_test")
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert 'load_split("test")' not in source
    assert 'load_split("official' not in source


def test_gpu_device_explicitness(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert str(runner.resolve_device("cuda")) == "cuda:1"
    assert str(runner.resolve_device("cpu")) == "cpu"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert str(runner.resolve_device("cuda")) == "cuda:0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError):
        runner.resolve_device("cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(RuntimeError):
        runner.resolve_device("cuda")


def test_interventions_semantics():
    _reference, purified = _models()
    batch = _synthetic_batch(2)
    with torch.no_grad():
        clean = purified(batch, return_aux=True)[1]
        zeroed = purified(batch, interventions=pur.Interventions(coord_zero=True), return_aux=True)[1]
        node_off = purified(
            batch, interventions=pur.Interventions(constant_node_scale=0.0), return_aux=True
        )[1]
    assert torch.equal(zeroed["coord"], torch.zeros_like(zeroed["coord"]))
    assert not torch.equal(clean["E"], zeroed["E"])
    assert not torch.equal(clean["E"], node_off["E"])
    assert "constant_node_scale" in pur.Interventions.__dataclass_fields__
