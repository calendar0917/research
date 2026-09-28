"""Focused CPU tests for E2E-DictEnv-Clean-Mechanism-v1 (no GPU, no official test).

Covers the pre-registered test surface
(preregistration ``e2e_dictenv_clean_mechanism_v1`` section 14):

* BASE unchanged identity (``CleanMechModel`` == ``AuditModel``);
* C6 exact feature contract (canonical set == recorded mask, channels inert);
* node analytic independence null toy example;
* node n=0 / n=1 exact ``paired == indep``;
* node permutation invariance of the null;
* edge analytic independence null toy example;
* edge n=0 / n=1 / endpoint symmetry;
* relation coordinate groups and candidate provenance;
* DenseTied reproduces the frozen P1 definition;
* frozen gates;
* CPU-only guard and official-test blocker;
* matched-init contract for the new arms.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm

REPO_ROOT = Path(__file__).resolve().parents[3]
VALID_CACHE = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_p1/cache/env_valid.pt"


def _dictionary(seed: int = 0) -> np.ndarray:
    dictionary = np.random.RandomState(seed).randn(p2.PHI_DIM, 32).astype(np.float32)
    return dictionary / np.maximum(np.linalg.norm(dictionary, axis=0, keepdims=True), 1e-6)


def _synthetic(
    n_mols: int = 2, n_nodes: int = 4, seed: int = 7
) -> types.SimpleNamespace:
    """Batched collate-like input (same layout conventions as the audit tests)."""
    gen = torch.Generator().manual_seed(int(seed))
    nodes = int(n_mols) * int(n_nodes)
    occ_node, occ_root, occ_shell = [], [], []
    bond_root, bond_sp, bond_type, bond_u, bond_v = [], [], [], [], []
    batch: list[int] = []
    pair_sources: list[int] = []
    pair_targets: list[int] = []
    for molecule in range(int(n_mols)):
        offset = molecule * int(n_nodes)
        for root in range(int(n_nodes)):
            for shell in range(3):
                occ_root.append(offset + root)
                occ_node.append(offset + ((root + shell) % int(n_nodes)))
                occ_shell.append(shell)
            for shellpair in range(6):
                bond_root.append(offset + root)
                bond_sp.append(shellpair)
                bond_type.append(int(torch.randint(0, 4, (1,), generator=gen).item()))
                bond_u.append(offset + ((root + 1) % int(n_nodes)))
                bond_v.append(offset + ((root + 2) % int(n_nodes)))
        for left in range(int(n_nodes)):
            for right in range(left + 1, int(n_nodes)):
                pair_sources.append(offset + left)
                pair_targets.append(offset + right)
        batch.extend([molecule] * int(n_nodes))
    n_pairs = len(pair_sources)
    relation = torch.zeros((n_pairs, 23))
    bucket = torch.zeros((n_pairs,), dtype=torch.long)
    for index in range(n_pairs):
        distance = 1 + (index % 4)
        bucket[index] = distance - 1
        relation[index, distance - 1] = 1.0
        relation[index, 5] = float(np.log1p(distance))
        relation[index, 6:11] = torch.rand(5, generator=gen)
        relation[index, 11:14] = torch.rand(3, generator=gen)
        relation[index, 14:18] = torch.rand(4, generator=gen)
        relation[index, 18] = float(np.log1p(index % 5 + 1))
    data = types.SimpleNamespace(
        dict_phi=torch.rand((nodes, p2.PHI_DIM), generator=gen),
        dict_atom=torch.randint(0, p2.ATOM_CATEGORIES, (nodes,), generator=gen),
        anchor=torch.randn((nodes, audit.ANCHOR_DIM_EXPECTED), generator=gen),
        env_occ_node=torch.tensor(occ_node, dtype=torch.long),
        env_occ_root=torch.tensor(occ_root, dtype=torch.long),
        env_occ_shell=torch.tensor(occ_shell, dtype=torch.long),
        env_bond_root=torch.tensor(bond_root, dtype=torch.long),
        env_bond_shellpair=torch.tensor(bond_sp, dtype=torch.long),
        env_bond_type=torch.tensor(bond_type, dtype=torch.long),
        env_bond_u=torch.tensor(bond_u, dtype=torch.long),
        env_bond_v=torch.tensor(bond_v, dtype=torch.long),
        global_context=torch.randn((int(n_mols), audit.GLOBAL_DIM_EXPECTED), generator=gen),
        topology_features=torch.randn((int(n_mols), audit.TOPOLOGY_DIM_EXPECTED), generator=gen),
        pair_index=torch.tensor([pair_sources, pair_targets], dtype=torch.long),
        pair_relation=relation,
        pair_bucket=bucket,
        batch=torch.tensor(batch, dtype=torch.long),
        y=torch.randn((int(n_mols),), generator=gen),
    )
    return data


def _pair(model_a: cm.CleanMechModel, spec: cm.CleanMechSpec) -> cm.CleanMechModel:
    model_b = cm.build_clean_mech_model(_dictionary(), seed=0, spec=spec)
    model_b.load_state_dict(model_a.state_dict())
    model_b.eval()
    return model_b


def _reference_model() -> cm.CleanMechModel:
    model = cm.build_clean_mech_model(_dictionary(), seed=0)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# BASE identity + C6 contract
# ---------------------------------------------------------------------------


def test_base_default_is_bit_identical_to_audit_model():
    reference = audit.build_audit_model(_dictionary(), seed=0)
    candidate = cm.build_clean_mech_model(_dictionary(), seed=0)
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    assert tuple(reference_state) == tuple(candidate_state)
    for key in reference_state:
        assert torch.equal(reference_state[key], candidate_state[key]), key
    reference.eval()
    candidate.eval()
    data = _synthetic()
    with torch.no_grad():
        assert torch.equal(reference(data, mask=None), candidate(data, mask=None))
        assert torch.equal(
            reference(data, mask=audit.candidate_mask("C6")),
            candidate(data, mask=cm.C6_MASK),
        )


def test_c6_mask_is_the_canonical_set_of_the_recorded_candidate():
    assert cm.c6_equivalence_check()
    recorded = audit.candidate_mask("C6")
    assert set(cm.C6_MASK.global_zero_groups) == {"atom_histogram", "bond_histogram"}
    assert set(recorded.global_zero_groups) == set(cm.C6_MASK.global_zero_groups)
    assert cm.C6_MASK.unary_zero_blocks == ("count",)
    assert cm.C6_MASK.pair_zero_blocks == ("count",)
    assert cm.C6_MASK.relation_zero_groups == ("path_count",)


def test_c6_removes_graph_chemistry_relation_path_count_and_counts():
    model = _reference_model()
    data = _synthetic()
    changed = types.SimpleNamespace(**data.__dict__)
    changed.global_context = data.global_context.clone()
    changed.pair_relation = data.pair_relation.clone()
    changed.global_context[:, 30:62] = torch.randn_like(changed.global_context[:, 30:62])
    changed.pair_relation[:, 18] = torch.randn_like(changed.pair_relation[:, 18])
    with torch.no_grad():
        base = model(data, mask=cm.C6_MASK)
        perturbed = model(changed, mask=cm.C6_MASK)
        identity_base = model(data, mask=None)
        identity_perturbed = model(changed, mask=None)
    # C6 forward is bit-identical under the removed channels; the untouched
    # identity forward changes.
    assert torch.equal(base, perturbed)
    assert not torch.equal(identity_base, identity_perturbed)
    # count blocks are structurally zero under C6 pooling.
    unary = audit.pool_moments_masked(
        torch.ones((5, p2.ENV_DIM)), torch.tensor([0, 0, 1, 1, 1]), 2, ("count",)
    )
    assert float(unary[:, audit.UNARY_BLOCKS["count"][0] :].abs().max()) == 0.0
    assert float(unary[:, : audit.UNARY_BLOCKS["count"][0]].sum()) > 0.0
    pair = audit.pool_pair_moments_masked(
        torch.ones((5, p2.PAIR_HIDDEN)),
        torch.tensor([0, 0, 1, 1, 1]),
        torch.tensor([0, 0, 1, 1, 1]),
        2,
        ("count",),
    )
    view = pair.reshape(2, p2.DISTANCE_BUCKETS, audit.PAIR_BLOCK_DIM)
    assert float(view[:, :, audit.PAIR_BLOCKS["count"][0] :].abs().max()) == 0.0


# ---------------------------------------------------------------------------
# node analytic independence null
# ---------------------------------------------------------------------------


def _toy_node_data() -> tuple[types.SimpleNamespace, torch.Tensor]:
    """One root, two shells: shell0 has 1 occurrence, shell1 has 3."""
    coord = torch.randn((3, p1.K_ATOMS), generator=torch.Generator().manual_seed(11))
    data = types.SimpleNamespace(
        dict_atom=torch.tensor([0, 1, 2], dtype=torch.long),
        env_occ_node=torch.tensor([1, 2, 0, 0, 1, 2], dtype=torch.long),
        env_occ_root=torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.long),
        env_occ_shell=torch.tensor([1, 1, 1, 0, 2, 2], dtype=torch.long),
        env_bond_root=torch.tensor([], dtype=torch.long),
        env_bond_shellpair=torch.tensor([], dtype=torch.long),
        env_bond_type=torch.tensor([], dtype=torch.long),
        env_bond_u=torch.tensor([], dtype=torch.long),
        env_bond_v=torch.tensor([], dtype=torch.long),
    )
    return data, coord


def test_node_independence_null_toy_example():
    model = _reference_model().double()
    data, coord = _toy_node_data()
    coord = coord.double()
    with torch.no_grad():
        paired = model.node_slots(coord, data, binding="paired")
        indep = model.node_slots(coord, data, binding="indep")
        # shell 0: single occurrence -> exact equality
        assert torch.equal(paired[:, 0], indep[:, 0])
        # shell 1: n = 3; explicit analytic reference in float64
        q = torch.nn.functional.one_hot(data.dict_atom, p2.ATOM_CATEGORIES).double()
        occ = data.env_occ_node
        a = coord[occ] @ model.W_A_S.double()
        c = q[occ] @ model.W_A_C.double()
        mask = data.env_occ_shell == 1
        expected_pair = (a[mask] * c[mask]).sum(0) / np.sqrt(p2.D_A)
        expected_indep = (a[mask].sum(0) * c[mask].sum(0)) / (3.0 * np.sqrt(p2.D_A))
        assert torch.allclose(paired[0, 1], expected_pair, atol=1e-8)
        assert torch.allclose(indep[0, 1], expected_indep, atol=1e-8)
        # the residual is nonzero for a generic assignment
        assert float((paired[0, 1] - indep[0, 1]).abs().max()) > 1e-6


def test_node_independence_null_zero_and_one_slots():
    model = _reference_model().double()
    data, coord = _toy_node_data()
    coord = coord.double()
    with torch.no_grad():
        paired = model.node_slots(coord, data, binding="paired")
        indep = model.node_slots(coord, data, binding="indep")
    # empty data (no occurrences at all): all slots zero
    empty = types.SimpleNamespace(
        dict_atom=torch.tensor([0], dtype=torch.long),
        env_occ_node=torch.tensor([], dtype=torch.long),
        env_occ_root=torch.tensor([], dtype=torch.long),
        env_occ_shell=torch.tensor([], dtype=torch.long),
        env_bond_root=torch.tensor([], dtype=torch.long),
        env_bond_shellpair=torch.tensor([], dtype=torch.long),
        env_bond_type=torch.tensor([], dtype=torch.long),
        env_bond_u=torch.tensor([], dtype=torch.long),
        env_bond_v=torch.tensor([], dtype=torch.long),
    )
    with torch.no_grad():
        empty_slots = model.node_slots(coord[:1], empty, binding="indep")
    assert float(empty_slots.abs().max()) == 0.0
    # n = 1 slots are exactly paired (shell 2 has two occurrences here; build a
    # single-occurrence shell by keeping only the first shell-2 row)
    single = types.SimpleNamespace(**{**data.__dict__})
    single.env_occ_node = torch.tensor([0, 0, 0, 0, 1], dtype=torch.long)
    single.env_occ_root = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    single.env_occ_shell = torch.tensor([1, 1, 1, 0, 2], dtype=torch.long)
    with torch.no_grad():
        paired_single = model.node_slots(coord, single, binding="paired")
        indep_single = model.node_slots(coord, single, binding="indep")
    assert torch.equal(paired_single[:, 2], indep_single[:, 2])
    assert torch.equal(paired_single[:, 0], indep_single[:, 0])


def test_node_independence_null_is_assignment_invariant():
    model = _reference_model().double()
    data, coord = _toy_node_data()
    coord = coord.double()
    q = torch.nn.functional.one_hot(data.dict_atom, p2.ATOM_CATEGORIES).double()
    occ = data.env_occ_node
    mask = data.env_occ_shell == 1
    a = (coord[occ][mask] @ model.W_A_S.double())
    c = (q[occ][mask] @ model.W_A_C.double())
    a_perm = a[[2, 0, 1]]  # break the alpha <-> q correspondence only
    paired = (a * c).sum(0) / np.sqrt(p2.D_A)
    paired_perm = (a_perm * c).sum(0) / np.sqrt(p2.D_A)
    indep = (a.sum(0) * c.sum(0)) / (3.0 * np.sqrt(p2.D_A))
    assert not torch.allclose(paired, paired_perm, atol=1e-6), "correspondence probe is inert"
    assert torch.allclose((a_perm.sum(0) * c.sum(0)) / (3.0 * np.sqrt(p2.D_A)), indep, atol=1e-6)
    # and the module operator reproduces both identities
    with torch.no_grad():
        module_paired = model.node_slots(coord, data, binding="paired")[0, 1]
        module_indep = model.node_slots(coord, data, binding="indep")[0, 1]
    assert torch.allclose(module_paired, paired, atol=1e-8)
    assert torch.allclose(module_indep, indep, atol=1e-8)


# ---------------------------------------------------------------------------
# edge analytic independence null
# ---------------------------------------------------------------------------


def _toy_edge_data() -> tuple[types.SimpleNamespace, torch.Tensor]:
    coord = torch.randn((4, p1.K_ATOMS), generator=torch.Generator().manual_seed(13))
    data = types.SimpleNamespace(
        env_bond_root=torch.tensor([0, 0, 0, 0, 0], dtype=torch.long),
        env_bond_shellpair=torch.tensor([1, 1, 1, 0, 2], dtype=torch.long),
        env_bond_type=torch.tensor([0, 1, 2, 3, 0], dtype=torch.long),
        env_bond_u=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
        env_bond_v=torch.tensor([1, 2, 3, 1, 2], dtype=torch.long),
    )
    return data, coord


def test_edge_independence_null_toy_example():
    model = _reference_model().double()
    data, coord = _toy_edge_data()
    coord = coord.double()
    with torch.no_grad():
        paired = model.edge_slots(coord, data, binding="paired")
        indep = model.edge_slots(coord, data, binding="indep")
        cu = coord[data.env_bond_u]
        cv = coord[data.env_bond_v]
        g = torch.cat([cu + cv, (cu - cv).abs(), cu * cv], dim=1)
        s = g @ model.W_E_S.double()
        b = torch.nn.functional.one_hot(data.env_bond_type, p2.BOND_CATEGORIES).double()
        c = b @ model.W_E_C.double()
        mask = data.env_bond_shellpair == 1
        expected_pair = (s[mask] * c[mask]).sum(0) / np.sqrt(48)
        expected_indep = (s[mask].sum(0) * c[mask].sum(0)) / (3.0 * np.sqrt(48))
        assert torch.allclose(paired[0, 1], expected_pair, atol=1e-8)
        assert torch.allclose(indep[0, 1], expected_indep, atol=1e-8)
        # shellpair 0 has one bond -> exact equality; empty shellpair -> zero
        assert torch.equal(paired[0, 0], indep[0, 0])
        assert float(indep[0, 3:].abs().max()) == 0.0
        assert float(paired[0, 3:].abs().max()) == 0.0


def test_edge_independence_null_endpoint_symmetry():
    model = _reference_model().double()
    data, coord = _toy_edge_data()
    coord = coord.double()
    swapped = types.SimpleNamespace(**{**data.__dict__})
    swapped.env_bond_u = data.env_bond_v
    swapped.env_bond_v = data.env_bond_u
    with torch.no_grad():
        indep = model.edge_slots(coord, data, binding="indep")
        indep_swapped = model.edge_slots(coord, swapped, binding="indep")
        paired = model.edge_slots(coord, data, binding="paired")
        paired_swapped = model.edge_slots(coord, swapped, binding="paired")
    assert torch.allclose(indep, indep_swapped, atol=1e-7)
    assert torch.allclose(paired, paired_swapped, atol=1e-7)


# ---------------------------------------------------------------------------
# relation candidates
# ---------------------------------------------------------------------------


def test_relation_candidates_only_read_declared_coordinates():
    relation = torch.randn((4, 15))
    for name, mask in cm.RELATION_CANDIDATES.items():
        assert set(mask.relation_zero_groups) <= set(audit.RELATION_GROUPS), name
        # full (C6) only removes the already-removed path count
        if name == "REL-FULL-CLEAN":
            assert set(mask.relation_zero_groups) == {"path_count"}
        if name == "REL-DIST-BOUNDARY":
            assert set(mask.relation_zero_groups) == {"overlap", "path_count"}
        if name == "REL-DIST":
            assert set(mask.relation_zero_groups) == {"boundary", "overlap", "path_count"}
        out = audit._replace_grouped_columns(
            relation, audit.RELATION_GROUPS, mask.relation_zero_groups
        )
        for group, (low, high) in audit.RELATION_GROUPS.items():
            if group in mask.relation_zero_groups:
                assert float(out[:, low:high].abs().max()) == 0.0
            else:
                assert torch.equal(out[:, low:high], relation[:, low:high])


def test_relation_group_provenance_matches_the_p1_used_slice():
    assert p1.P1_RELATION_INDICES == tuple(range(14)) + (18,)
    assert audit.RELATION_GROUPS["distance"] == (0, 6)
    assert audit.RELATION_GROUPS["overlap"] == (6, 11)
    assert audit.RELATION_GROUPS["boundary"] == (11, 14)
    assert audit.RELATION_GROUPS["path_count"] == (14, 15)
    assert sum(high - low for low, high in audit.RELATION_GROUPS.values()) == 15


# ---------------------------------------------------------------------------
# DenseTied control
# ---------------------------------------------------------------------------


def test_dense_tied_reproduces_the_p1_dense_operator():
    dictionary = _dictionary(seed=5)
    model = cm.build_clean_mech_model(
        dictionary, seed=0, spec=cm.make_spec("dense", mask_kind="C6", coding="dense_tied")
    )
    phi = torch.randn((7, p2.PHI_DIM), generator=torch.Generator().manual_seed(17))
    with torch.no_grad():
        code = model.code(phi)
        expected = phi @ v0.normalized_dictionary(model.D)
    assert torch.allclose(code, expected, atol=1e-6)
    assert int(sum(parameter.numel() for parameter in model.parameters())) == 97487
    sparse = cm.build_clean_mech_model(dictionary, seed=0)
    assert int(sum(parameter.numel() for parameter in sparse.parameters())) == int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    assert tuple(sparse.state_dict()) == tuple(model.state_dict())


def test_dense_tied_shares_initialisation_with_sparse():
    dictionary = _dictionary(seed=6)
    sparse = cm.build_clean_mech_model(dictionary, seed=4)
    dense = cm.build_clean_mech_model(
        dictionary, seed=4, spec=cm.make_spec("dense", mask_kind="C6", coding="dense_tied")
    )
    for key, value in sparse.state_dict().items():
        assert torch.equal(value, dense.state_dict()[key]), key


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def test_c6_gate_thresholds():
    strong = cm.c6_gate({0: -0.0148, 1: -0.006, 2: -0.004})
    assert strong["verdict"] == "STRONG_SUPPORT" and strong["adopted"]
    clean = cm.c6_gate({0: 0.001, 1: -0.002, 2: 0.002})
    assert clean["verdict"] == "CLEANNESS_SUPPORT" and clean["adopted"]
    unstable = cm.c6_gate({0: -0.010, 1: 0.010, 2: 0.0})
    assert unstable["verdict"] == "UNSTABLE" and not unstable["adopted"]
    reject = cm.c6_gate({0: 0.006, 1: 0.008, 2: 0.005})
    assert reject["verdict"] == "REJECT" and not reject["adopted"]
    mixed = cm.c6_gate({0: 0.002, 1: 0.002, 2: 0.002})
    assert mixed["verdict"] == "CLEANNESS_SUPPORT"


def test_independence_relation_and_specificity_gates():
    node = cm.independence_gate("node", {0: 0.001})
    assert node["verdict"] == "EXTEND_SEEDS"
    node_stop = cm.independence_gate("node", {0: 0.020})
    assert node_stop["verdict"] == "STOP_MATERIALLY_NEEDED"
    edge = cm.independence_gate("edge", {0: 0.100})
    assert edge["verdict"] == "STOP_EDGE_ASSIGNMENT_NEEDED"
    edge_close = cm.independence_gate("edge", {0: 0.008})
    assert edge_close["verdict"] == "EXTEND_SEEDS"
    assert cm.relation_gate(0.004)["verdict"] == "EXTEND_SEEDS"
    assert cm.relation_gate(0.020)["verdict"] == "STOP_TOO_BIG"
    assert cm.relation_gate(0.010)["verdict"] == "INCONCLUSIVE_KEEP_FULL"
    assert cm.specificity_gate(0.004)["verdict"] == "SPARSE_SPECIFIC_CANDIDATE"
    assert cm.specificity_gate(-0.002)["verdict"] == "SPECIFICITY_NOT_ESTABLISHED"


# ---------------------------------------------------------------------------
# matched-init contract, CPU guard, official-test blocker
# ---------------------------------------------------------------------------


def _state_hash(model: torch.nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_new_arms_share_initialisation_and_parameter_shapes():
    dictionary = _dictionary(seed=9)
    hashes = {}
    for tag in ("C6", "C6-NODE-INDEP", "C6-EDGE-INDEP", "C6-BOTH-INDEP", "C6-DENSE-TIED"):
        model = cm.build_clean_mech_model(dictionary, seed=0, spec=cm.CLEAN_MECH_ARMS[tag])
        hashes[tag] = _state_hash(model)
        assert int(sum(p.numel() for p in model.parameters())) == 97487
    assert len(set(hashes.values())) == 1, hashes


def test_train_factory_default_is_the_frozen_build_path():
    import inspect

    signature = inspect.signature(audit.train_cpu)
    assert signature.parameters["model_factory"].default is None
    assert signature.parameters["arm_spec"].default is None
    dictionary = _dictionary(seed=3)
    default = audit.build_audit_model(dictionary, seed=2)
    via_factory = (lambda d, s: audit.build_audit_model(d, s))(dictionary, 2)
    for key, value in default.state_dict().items():
        assert torch.equal(value, via_factory.state_dict()[key]), key


def test_cpu_only_guard():
    model = _reference_model()
    with pytest.raises(RuntimeError):
        cm.independence_norm_stats(model, [], torch.device("cuda"))
    with pytest.raises(RuntimeError):
        cm.dictionary_diagnostics(model, [], torch.device("cuda"), _dictionary())


def test_module_never_touches_official_test_or_accelerators():
    source = Path(cm.__file__).read_text(encoding="utf-8")
    for token in ("cuda", "CUDA_VISIBLE", "official_test_loaded = True"):
        assert token not in source, token
    for spec in cm.CLEAN_MECH_ARMS.values():
        assert spec.node_binding in cm.NODE_BINDINGS
        assert spec.edge_binding in cm.EDGE_BINDINGS
        assert spec.coding in cm.CODINGS


def test_find_artifact_resolves_seed_suffixed_runs(tmp_path, monkeypatch):
    """Regression: formal runs live in ``<tag>_seed<k>_e<epochs>.json`` files."""
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_clean_mechanism_v1 as runner

    stage_dirs = {key: tmp_path / key for key in ("a", "c", "d", "f")}
    for path in stage_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(runner, "STAGE_DIRS", stage_dirs)
    monkeypatch.setattr(runner, "REUSED", {})
    payload = {
        "soup": {"soup_valid_mae": 0.5},
        "arm_spec": cm.CLEAN_MECH_ARMS["C6"].as_dict(),
    }
    (stage_dirs["a"] / "C6_seed1_e320.json").write_text(json.dumps(payload), encoding="utf-8")
    (stage_dirs["a"] / "C6_seed1_soup_state.pt").write_text("stub", encoding="utf-8")
    entry = runner.find_artifact("C6", 1)
    assert entry is not None
    assert entry["source"] == "a"
    assert runner.soup_mae(entry) == pytest.approx(0.5)
    assert entry["soup_state"].name == "C6_seed1_soup_state.pt"
    assert runner.find_artifact("C6", 2) is None
    assert runner.find_artifact("C1", 1) is None


@pytest.mark.skipif(not VALID_CACHE.exists(), reason="P1 valid cache not available locally")
def test_independence_norm_stats_and_binding_variants_on_real_data():
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    device = audit.attach_cpu(2)
    dictionary, _sha = p2run.load_dictionary("sdb32")
    graphs = p1run.load_split("valid", subset=8)
    loader = p1.make_env_loader(graphs, 4, False, 0)
    model = cm.build_clean_mech_model(dictionary, seed=0)
    stats = cm.independence_norm_stats(model, loader, device)
    assert stats["node"]["shell0"]["n_one"] == stats["node"]["shell0"]["n_slots"]
    assert stats["node"]["shell0"]["residual_norm_mean"] == 0.0
    assert stats["node"]["shell1"]["n_multi"] > 0
    assert "shellpair0" in stats["edge"]
    state = model.state_dict()
    with torch.no_grad():
        baseline = audit.evaluate_mask(model, loader, device, cm.C6_MASK)
    variant = cm.evaluate_binding_variant(
        dictionary,
        state,
        loader,
        device,
        cm.C6_MASK,
        node_binding="indep",
        edge_binding="paired",
        baseline_predictions=baseline["predictions"],
    )
    assert variant["valid_mae"] > 0.0
    assert variant["mean_abs_prediction_delta"] >= 0.0
    assert variant["official_test_loaded"] is False
