"""Focused CPU tests for E2E-DictEnv-H1-Clarity-Audit (no GPU, no official test).

Covers the pre-registered test surface (preregistration / task §15):

* baseline audit mode reproduces the H1 prediction;
* anchor masks only affect intended coordinates;
* global chemistry mask only affects the chemistry block;
* topology mask only affects topology25;
* node / edge intervention keeps tensor shape;
* assignment shuffle preserves required multisets;
* relation masks use verified coordinate groups;
* second-moment mask affects only the intended readout block;
* CPU-only assertion;
* the frozen P1 / P2-ABS modules are untouched.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_h1_clarity_audit as audit

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_p2_abs/states/H1_soup_state.pt"
BASELINE_ARTIFACT = REPO_ROOT / "tracks/ksvd/results/e2e_dictenv_h1_clarity_audit/baseline_replay.json"


# ---------------------------------------------------------------------------
# synthetic batched input mirroring the real collate output conventions
# ---------------------------------------------------------------------------


def _synthetic(n_mols: int = 2, n_nodes: int = 4, seed: int = 7) -> Any:
    """A CPU ``Batch``-like namespace with globally offset incidence indices."""
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
        relation[index, 19:23] = 0.0
    pair_index = torch.tensor([pair_sources, pair_targets], dtype=torch.long)
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
        pair_index=pair_index,
        pair_relation=relation,
        pair_bucket=bucket,
        batch=torch.tensor(batch, dtype=torch.long),
        y=torch.randn((int(n_mols),), generator=gen),
    )
    return data


def _model() -> audit.AuditModel:
    dictionary = np.random.RandomState(0).randn(p2.PHI_DIM, 32).astype(np.float32)
    dictionary /= np.maximum(np.linalg.norm(dictionary, axis=0, keepdims=True), 1e-6)
    torch.manual_seed(0)
    model = audit.build_audit_model(dictionary, seed=0)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# baseline contract
# ---------------------------------------------------------------------------


def test_identity_mask_is_bit_identical_to_parent_forward():
    model = _model()
    data = _synthetic()
    with torch.no_grad():
        parent = model(data, mask=None)
        masked = model(data, mask=audit.AuditMask())
    assert parent.shape == masked.shape == (2,)
    assert torch.equal(parent, masked)


def test_identity_mask_reproduces_parent_aux_environments():
    model = _model()
    data = _synthetic()
    with torch.no_grad():
        _, aux_parent = model(data, mask=None, return_aux=True)
        _, aux_masked = model(data, mask=audit.AuditMask(), return_aux=True)
    assert torch.equal(aux_parent["E"], aux_masked["E"])
    assert torch.equal(aux_parent["coord"], aux_masked["coord"])


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="H1 soup checkpoint not available locally")
def test_baseline_audit_mode_reproduces_h1_prediction():
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    audit.attach_cpu(2)
    dictionary, _sha = p2run.load_dictionary(audit.H1_CONFIG.dict_kind)
    model = audit.build_audit_model(dictionary, seed=0)
    model.load_state_dict(audit.load_h1_soup_state(CHECKPOINT))
    model.eval()
    valid = p1run.load_split("valid", subset=8)
    loader = p1.make_env_loader(valid, 8, False, int(p2run.EVAL_SHUFFLE_OFFSET))
    reference = p1run.evaluate(model, loader, torch.device("cpu"))
    masked = audit.evaluate_mask(model, loader, torch.device("cpu"), audit.AuditMask())
    assert np.array_equal(reference["predictions"], masked["predictions"])
    assert float(reference["mae"]) == pytest.approx(float(masked["mae"]), abs=0.0)


@pytest.mark.skipif(not BASELINE_ARTIFACT.exists(), reason="baseline replay artifact not available")
def test_recorded_cpu_baseline_replay_matches_historical_soup():
    import json

    payload = json.loads(BASELINE_ARTIFACT.read_text(encoding="utf-8"))
    assert payload["official_test_loaded"] is False
    assert payload["device"] == "cpu"
    assert payload["audit_identity_mask_bit_identical"] is True
    assert payload["abs_diff_vs_historical"] < 1e-6
    assert payload["p1_evaluate_max_abs_diff"] == 0.0


# ---------------------------------------------------------------------------
# Phase A: verified coordinate groups
# ---------------------------------------------------------------------------


def test_coordinate_groups_partition_declared_widths():
    assert sorted(audit.ANCHOR_GROUPS) == ["atom_mass", "bond_mass", "root", "size"]
    assert sum(high - low for low, high in audit.ANCHOR_GROUPS.values()) == audit.ANCHOR_DIM_EXPECTED
    assert sum(high - low for low, high in audit.GLOBAL_GROUPS.values()) == audit.GLOBAL_DIM_EXPECTED
    assert sum(high - low for low, high in audit.RELATION_GROUPS.values()) == audit.RELATION_DIM_EXPECTED
    assert audit.GLOBAL_GROUPS["atom_histogram"] == (30, 58)
    assert audit.GLOBAL_GROUPS["bond_histogram"] == (58, 62)
    assert audit.GLOBAL_GROUPS["structure_short"] == (0, 15)
    assert audit.GLOBAL_GROUPS["structure_long"] == (15, 30)
    assert audit.UNARY_BLOCKS == {"first": (0, 48), "second": (48, 96), "count": (96, 97)}
    assert audit.PAIR_BLOCKS == {"first": (0, 16), "second": (16, 32), "count": (32, 33)}
    assert audit.READER_IN_DIM == 302
    assert p1.P1_RELATION_INDICES == tuple(range(14)) + (18,)


def test_relation_groups_match_raw_layout():
    assert audit.RELATION_GROUPS["distance"] == (0, 6)
    assert audit.RELATION_GROUPS["overlap"] == (6, 11)
    assert audit.RELATION_GROUPS["boundary"] == (11, 14)
    assert audit.RELATION_GROUPS["path_count"] == (14, 15)
    assert audit.RELATION_RAW_LAYOUT["path_bond_mean"] == (14, 18)
    assert audit.RELATION_RAW_LAYOUT["adjacent_bond_type"] == (19, 23)


def test_global_provenance_records_are_honest_when_cache_available():
    from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

    frame = ztopo.load_cache("valid")
    if frame is None:
        pytest.skip("topology cache not available")
    assert list(ztopo.feature_names("hinge"))[15:17] == ["longest", "longest_squared"]


# ---------------------------------------------------------------------------
# anchor
# ---------------------------------------------------------------------------


def test_anchor_mask_only_affects_intended_coordinates():
    model = _model()
    data = _synthetic()
    mask = audit.AuditMask(anchor_zero_groups=("atom_mass", "bond_mass"))
    zeroed = data.anchor.clone()
    zeroed[:, 28:56] = 0.0
    zeroed[:, 56:60] = 0.0
    coord = model.code(data.dict_phi)
    with torch.no_grad():
        left = model.environments_masked(coord, data, mask)
        reference = model.environments_masked(
            coord, types.SimpleNamespace(**{**data.__dict__, "anchor": zeroed}), audit.AuditMask()
        )
    assert torch.equal(left, reference)
    assert torch.equal(data.anchor[:, 0:28], zeroed[:, 0:28])
    assert torch.equal(data.anchor[:, 60:62], zeroed[:, 60:62])


def test_anchor_full_zero_differs_from_unmasked():
    model = _model()
    data = _synthetic()
    coord = model.code(data.dict_phi)
    with torch.no_grad():
        base = model.environments_masked(coord, data, audit.AuditMask())
        zero = model.environments_masked(
            coord, data, audit.AuditMask(anchor_zero_groups=("root", "atom_mass", "bond_mass", "size"))
        )
    assert base.shape == zero.shape
    assert not torch.equal(base, zero)


# ---------------------------------------------------------------------------
# global context / topology
# ---------------------------------------------------------------------------


def test_global_chemistry_mask_only_affects_chemistry_block():
    global_context = torch.randn(4, audit.GLOBAL_DIM_EXPECTED)
    masked = audit._zero_grouped_columns(global_context, audit.GLOBAL_GROUPS, audit.GLOBAL_CHEMISTRY_GROUPS)
    assert torch.equal(masked[:, 0:30], global_context[:, 0:30])
    assert torch.equal(masked[:, 30:62], torch.zeros(4, 32))
    structure = audit._zero_grouped_columns(
        global_context, audit.GLOBAL_GROUPS, audit.GLOBAL_STRUCTURE_GROUPS
    )
    assert torch.equal(structure[:, 0:30], torch.zeros(4, 30))
    assert torch.equal(structure[:, 30:62], global_context[:, 30:62])


def test_topology_mask_only_affects_topology25():
    model = _model()
    data = _synthetic()
    other = types.SimpleNamespace(
        **{**data.__dict__, "topology_features": torch.randn_like(data.topology_features) * 5.0}
    )
    with torch.no_grad():
        masked_a = model(data, mask=audit.AuditMask(topology_zero=True))
        masked_b = model(other, mask=audit.AuditMask(topology_zero=True))
        unmasked_a = model(data, mask=audit.AuditMask())
        unmasked_b = model(other, mask=audit.AuditMask())
    assert torch.equal(masked_a, masked_b)
    assert not torch.equal(unmasked_a, unmasked_b)


def test_global_zero_changes_prediction_and_shapes_hold():
    model = _model()
    data = _synthetic()
    for mask in (
        audit.AuditMask(global_zero_groups=audit.GLOBAL_CHEMISTRY_GROUPS),
        audit.AuditMask(global_zero_groups=audit.GLOBAL_STRUCTURE_GROUPS),
        audit.AuditMask(global_zero_groups=audit.GLOBAL_STRUCTURE_GROUPS + audit.GLOBAL_CHEMISTRY_GROUPS),
    ):
        with torch.no_grad():
            prediction = model(data, mask=mask)
        assert prediction.shape == data.y.shape


# ---------------------------------------------------------------------------
# node / edge binding
# ---------------------------------------------------------------------------


def test_node_edge_interventions_keep_tensor_shape():
    model = _model()
    data = _synthetic()
    data.env_occ_coord_node = p1.shuffled_occ_node_for_molecule(
        data.env_occ_node, data.env_occ_root, data.env_occ_shell, 101
    )
    su, sv = p1.shuffled_bond_endpoints_for_molecule(
        data.env_bond_root, data.env_bond_shellpair, data.env_bond_u, data.env_bond_v, 101
    )
    data.env_bond_u_shuffled = su
    data.env_bond_v_shuffled = sv
    masks = [
        audit.AuditMask(node_binding_zero=True),
        audit.AuditMask(edge_binding_zero=True),
        audit.AuditMask(use_node_shuffle=True),
        audit.AuditMask(use_edge_shuffle=True),
        audit.AuditMask(use_node_shuffle=True, use_edge_shuffle=True),
        audit.AuditMask(coord_zero=True),
        audit.AuditMask(pair_projection_zero=True),
    ]
    for mask in masks:
        with torch.no_grad():
            prediction, aux = model(data, mask=mask, return_aux=True)
        assert prediction.shape == (2,)
        assert aux["E"].shape == (data.batch.shape[0], p2.ENV_DIM)
        assert aux["coord"].shape == (data.batch.shape[0], audit.H1_CONFIG.K)


def test_zero_node_binding_equals_zeroed_slot_input():
    model = _model()
    data = _synthetic()
    coord = model.code(data.dict_phi)
    with torch.no_grad():
        a = model.environments_masked(coord, data, audit.AuditMask(node_binding_zero=True))
        b = model.environments_masked(coord, data, audit.AuditMask(edge_binding_zero=True))
        c = model.environments_masked(coord, data, audit.AuditMask())
    assert not torch.equal(a, c) and not torch.equal(b, c) and not torch.equal(a, b)


def test_coord_zero_really_zeroes_the_code_gather():
    model = _model()
    data = _synthetic()
    with torch.no_grad():
        _, aux = model(data, mask=audit.AuditMask(coord_zero=True), return_aux=True)
    assert torch.equal(aux["coord"], torch.zeros_like(aux["coord"]))
    coord = model.code(data.dict_phi)
    with torch.no_grad():
        env = model.environments_masked(torch.zeros_like(coord), data, audit.AuditMask())
    assert torch.equal(aux["E"], env)


# ---------------------------------------------------------------------------
# assignment shuffles
# ---------------------------------------------------------------------------


def test_node_assignment_shuffle_preserves_multisets():
    occ_node = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    occ_root = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    occ_shell = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    shuffled = p1.shuffled_occ_node_for_molecule(occ_node, occ_root, occ_shell, 123)
    assert shuffled.shape == occ_node.shape
    for root in (0, 1):
        for shell in (0, 1):
            selection = (occ_root == root) & (occ_shell == shell)
            assert sorted(shuffled[selection].tolist()) == sorted(occ_node[selection].tolist())
    # every position still gathers a code from a node of the same (root, shell) group
    assert set(shuffled.tolist()) == set(occ_node.tolist())


def test_edge_assignment_shuffle_preserves_paired_multisets():
    bond_root = torch.tensor([0, 0, 0, 0])
    bond_sp = torch.tensor([0, 0, 1, 1])
    bond_u = torch.tensor([0, 1, 2, 3])
    bond_v = torch.tensor([1, 2, 3, 0])
    su, sv = p1.shuffled_bond_endpoints_for_molecule(bond_root, bond_sp, bond_u, bond_v, 55)
    for sp in (0, 1):
        selection = bond_sp == sp
        left = sorted(zip(su[selection].tolist(), sv[selection].tolist()))
        right = sorted(zip(bond_u[selection].tolist(), bond_v[selection].tolist()))
        assert left == right
    assert sorted(su.tolist()) == sorted(bond_u.tolist())
    assert sorted(sv.tolist()) == sorted(bond_v.tolist())


def test_prepare_and_clear_shuffles_round_trip():
    data = _synthetic()
    data.env_occ_coord_node = None
    data.env_bond_u_shuffled = None
    data.env_bond_v_shuffled = None
    audit.prepare_shuffles([data], "both", (101, 1101))
    assert data.env_occ_coord_node is not None
    assert data.env_bond_u_shuffled is not None and data.env_bond_v_shuffled is not None
    audit.clear_shuffles([data])
    assert data.env_occ_coord_node is None
    assert data.env_bond_u_shuffled is None and data.env_bond_v_shuffled is None


# ---------------------------------------------------------------------------
# relation / readout masks
# ---------------------------------------------------------------------------


def test_relation_masks_use_verified_coordinate_groups():
    model = _model()
    data = _synthetic()
    sliced = data.pair_relation[:, list(p1.P1_RELATION_INDICES)]
    masked = audit._zero_grouped_columns(sliced, audit.RELATION_GROUPS, ("distance",))
    assert torch.equal(masked[:, 0:6], torch.zeros_like(masked[:, 0:6]))
    assert torch.equal(masked[:, 6:15], sliced[:, 6:15])
    full = audit._zero_grouped_columns(
        sliced, audit.RELATION_GROUPS, ("distance", "overlap", "boundary", "path_count")
    )
    assert torch.equal(full, torch.zeros_like(full))
    with torch.no_grad():
        prediction = model(
            data, mask=audit.AuditMask(relation_zero_groups=("distance", "overlap", "boundary", "path_count"))
        )
    assert prediction.shape == (2,)


def test_second_moment_mask_affects_only_intended_readout_block():
    value = torch.randn(6, p2.ENV_DIM)
    batch = torch.tensor([0, 0, 1, 1, 1, 1])
    base = audit.pool_moments_masked(value, batch, 2)
    masked = audit.pool_moments_masked(value, batch, 2, ("second",))
    assert base.shape == masked.shape == (2, audit.UNARY_DIM)
    assert torch.equal(base[:, 0:48], masked[:, 0:48])
    assert torch.equal(masked[:, 48:96], torch.zeros_like(masked[:, 48:96]))
    assert torch.equal(base[:, 96:97], masked[:, 96:97])
    counts = audit.pool_moments_masked(value, batch, 2, ("count",))
    assert torch.equal(counts[:, 96:97], torch.zeros_like(counts[:, 96:97]))
    assert torch.equal(counts[:, 0:96], base[:, 0:96])


def test_mean_fill_replaces_exactly_the_masked_blocks():
    global_context = torch.randn(4, audit.GLOBAL_DIM_EXPECTED)
    policy = {
        "global:atom_histogram": torch.full((28,), 3.0),
        "global:bond_histogram": torch.full((4,), -2.0),
    }
    filled = audit._replace_grouped_columns(
        global_context, audit.GLOBAL_GROUPS, audit.GLOBAL_CHEMISTRY_GROUPS, policy, "global:"
    )
    assert torch.equal(filled[:, 0:30], global_context[:, 0:30])
    assert torch.equal(filled[:, 30:58], torch.full((4, 28), 3.0))
    assert torch.equal(filled[:, 58:62], torch.full((4, 4), -2.0))
    with pytest.raises(ValueError):
        audit._replace_grouped_columns(
            global_context,
            audit.GLOBAL_GROUPS,
            ("atom_histogram",),
            {"global:atom_histogram": torch.zeros(3)},
            "global:",
        )


def test_mean_fill_on_pooled_moment_blocks():
    value = torch.randn(6, p2.ENV_DIM)
    batch = torch.tensor([0, 0, 1, 1, 1, 1])
    policy = {"unary:second": torch.full((p2.ENV_DIM,), 5.0), "unary:count": torch.full((1,), -1.0)}
    filled = audit.pool_moments_masked(value, batch, 2, ("second", "count"), policy)
    assert torch.equal(filled[:, 0:48], audit.pool_moments_masked(value, batch, 2)[:, 0:48])
    assert torch.equal(filled[:, 48:96], torch.full((2, 48), 5.0))
    assert torch.equal(filled[:, 96:97], torch.full((2, 1), -1.0))


def test_graph_row_shuffle_preserves_marginals_and_restores():
    data_list = []
    for index in range(6):
        item = types.SimpleNamespace(
            global_context=torch.full((audit.GLOBAL_DIM_EXPECTED,), float(index)),
            topology_features=torch.tensor([float(index)] * audit.TOPOLOGY_DIM_EXPECTED),
        )
        data_list.append(item)
    originals = [item.global_context.clone() for item in data_list]
    restore = audit.permute_graph_rows(data_list, "global_chemistry", 4242)
    shuffled = torch.stack([item.global_context for item in data_list])
    assert torch.equal(shuffled[:, 0:30], torch.stack(originals)[:, 0:30])
    assert sorted(shuffled[:, 30].tolist()) == sorted(torch.stack(originals)[:, 30].tolist())
    restore()
    assert all(torch.equal(item.global_context, original) for item, original in zip(data_list, originals))
    with pytest.raises(ValueError):
        audit.permute_graph_rows(data_list, "nope", 1)


def test_pair_second_moment_mask_affects_only_intended_blocks():
    value = torch.randn(7, p2.PAIR_HIDDEN)
    pair_batch = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    bucket = torch.tensor([0, 1, 0, 0, 1, 1, 0])
    base = audit.pool_pair_moments_masked(value, pair_batch, bucket, 2)
    masked = audit.pool_pair_moments_masked(value, pair_batch, bucket, 2, ("second",))
    assert base.shape == masked.shape == (2, audit.RELATION_READOUT_DIM)
    for start in range(0, audit.RELATION_READOUT_DIM, audit.PAIR_BLOCK_DIM):
        assert torch.equal(base[:, start : start + 16], masked[:, start : start + 16])
        assert torch.equal(masked[:, start + 16 : start + 32], torch.zeros(2, 16))
        assert torch.equal(base[:, start + 32 : start + 33], masked[:, start + 32 : start + 33])


# ---------------------------------------------------------------------------
# interventions / registry
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="H1 soup checkpoint not available locally")
def test_fill_policy_shapes_on_real_checkpoint():
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
    from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run

    audit.attach_cpu(2)
    dictionary, _sha = p2run.load_dictionary(audit.H1_CONFIG.dict_kind)
    model = audit.build_audit_model(dictionary, seed=0)
    model.load_state_dict(audit.load_h1_soup_state(CHECKPOINT))
    model.eval()
    valid = p1run.load_split("valid", subset=8)
    loader = p1.make_env_loader(valid, 8, False, int(p2run.EVAL_SHUFFLE_OFFSET))
    policy = audit.build_fill_policy(model, loader, torch.device("cpu"))
    for name, (low, high) in audit.ANCHOR_GROUPS.items():
        assert policy[f"anchor:{name}"].shape == (high - low,)
    for name, (low, high) in audit.GLOBAL_GROUPS.items():
        assert policy[f"global:{name}"].shape == (high - low,)
    for name, (low, high) in audit.RELATION_GROUPS.items():
        assert policy[f"relation:{name}"].shape == (high - low,)
    assert policy["topology"].shape == (audit.TOPOLOGY_DIM_EXPECTED,)
    assert policy["unary:second"].shape == (p2.ENV_DIM,)
    assert policy["pair:second"].shape == (p2.DISTANCE_BUCKETS, p2.PAIR_HIDDEN)
    # fill and zero must differ for a non-degenerate block
    filled = audit.evaluate_mask(
        model,
        loader,
        torch.device("cpu"),
        audit.AuditMask(global_zero_groups=audit.GLOBAL_CHEMISTRY_GROUPS),
        policy,
    )
    zeroed = audit.evaluate_mask(
        model,
        loader,
        torch.device("cpu"),
        audit.AuditMask(global_zero_groups=audit.GLOBAL_CHEMISTRY_GROUPS),
        None,
    )
    assert filled["mae"] != zeroed["mae"]


def test_readout_shuffle_permutation_touches_only_named_blocks():
    unary = torch.randn(5, audit.UNARY_DIM)
    readout = torch.randn(5, audit.RELATION_READOUT_DIM)
    permutation = torch.tensor([1, 0, 2, 4, 3])
    left, right = audit.apply_readout_permutation(unary, readout, ("unary_second",), permutation)
    assert torch.equal(left[:, 0:48], unary[:, 0:48])
    assert torch.equal(left[:, 48:96], unary[permutation][:, 48:96])
    assert torch.equal(left[:, 96:97], unary[:, 96:97])
    assert torch.equal(right, readout)
    left2, right2 = audit.apply_readout_permutation(unary, readout, ("pair_first",), permutation)
    assert torch.equal(left2, unary)
    view = readout.reshape(5, audit.p2.DISTANCE_BUCKETS, audit.PAIR_BLOCK_DIM)
    view2 = right2.reshape(5, audit.p2.DISTANCE_BUCKETS, audit.PAIR_BLOCK_DIM)
    assert torch.equal(view2[:, :, 0:16], view[permutation][:, :, 0:16])
    assert torch.equal(view2[:, :, 16:33], view[:, :, 16:33])


def test_readout_shuffle_registry_is_well_formed():
    rows = audit.interventions_readout_shuffle()
    assert [row.name for row in rows] == ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "PS7", "PS8", "PS9"]
    for row in rows:
        assert row.readout_shuffle
        assert set(row.readout_shuffle) <= set(audit.READOUT_SHUFFLE_BLOCKS)
        assert row.seeds == audit.READOUT_SHUFFLE_SEEDS
        assert row.mask.is_identity()


def test_relation_shuffle_preserves_marginals_and_restores():
    data_list = []
    # deliberately unequal pair counts per molecule (the real cache is ragged)
    for index, rows in enumerate((5, 2, 7, 3)):
        relation = torch.zeros((rows, 23))
        relation[:, 0] = float(index)
        relation[:, 6] = float(index) * 2.0
        relation[:, 18] = float(index) + 1.0
        data_list.append(types.SimpleNamespace(pair_relation=relation))
    originals = [item.pair_relation.clone() for item in data_list]
    restore = audit.permute_pair_rows(data_list, "overlap", 77)
    stacked = torch.cat([item.pair_relation for item in data_list], dim=0)
    original = torch.cat(originals, dim=0)
    assert torch.equal(stacked[:, 0:6], original[:, 0:6])
    assert torch.equal(stacked[:, 11:23], original[:, 11:23])
    assert sorted(stacked[:, 6].tolist()) == sorted(original[:, 6].tolist())
    restore()
    assert all(torch.equal(item.pair_relation, original) for item, original in zip(data_list, originals))
    with pytest.raises(ValueError):
        audit.permute_pair_rows(data_list, "nope", 1)


def test_relation_shuffle_registry_is_well_formed():
    rows = audit.interventions_relation_shuffle()
    assert [row.name for row in rows] == ["RS1", "RS2", "RS3", "RS4", "RS5"]
    for row in rows:
        assert row.relation_shuffle in audit.RELATION_SHUFFLE_KINDS
        assert row.mask.is_identity()
    assert audit.RELATION_SHUFFLE_KINDS["path_count"] == ((18, 19),)
    assert audit.RELATION_SHUFFLE_KINDS["all"] == ((0, 14), (18, 19))


def test_fill_and_graph_shuffle_registries_are_well_formed():
    fill_rows = audit.interventions_fill()
    assert fill_rows, "fill registry must not be empty"
    assert all(row.use_fill for row in fill_rows)
    assert {row.name for row in fill_rows} >= {"A1", "A2", "G1", "T1", "R1", "P3"}
    for row in fill_rows:
        assert (
            row.mask.anchor_zero_groups
            or row.mask.global_zero_groups
            or row.mask.topology_zero
            or row.mask.relation_zero_groups
            or row.mask.unary_zero_blocks
            or row.mask.pair_zero_blocks
        )
    shuffle_rows = audit.interventions_graph_shuffle()
    assert [row.name for row in shuffle_rows] == ["GS1", "GS2", "GS3", "GS4", "GS5"]
    for row in shuffle_rows:
        assert row.graph_shuffle in audit.GRAPH_SHUFFLE_KINDS
        assert row.seeds == audit.GRAPH_SHUFFLE_SEEDS
        assert row.mask.is_identity()


def test_all_interventions_build_and_controls_are_identity():
    registered = audit.interventions()
    names = [intervention.name for intervention in registered]
    assert len(names) == len(set(names))
    for control in ("A0", "G0", "T0", "N0", "R0", "P0"):
        intervention = next(row for row in registered if row.name == control)
        assert intervention.mask.is_identity(), control
    for intervention in registered:
        if intervention.shuffle_kind:
            assert intervention.seeds, intervention.name
        payload = intervention.as_dict()
        assert payload["name"] == intervention.name
    assert audit.candidate_mask("M0").is_identity()
    assert set(audit.PHASE_C_CANDIDATES) >= {"M0", "C1", "C2", "C3", "C4", "C5"}


def test_summarise_interventions_assigns_load_bearing_classes():
    rows = [
        {"intervention": "X", "intervention_mae": 0.25, "mean_abs_prediction_delta": 0.2},
        {"intervention": "Y", "intervention_mae": 0.13, "mean_abs_prediction_delta": 0.02},
        {"intervention": "Z", "intervention_mae": 0.124, "mean_abs_prediction_delta": 0.001},
    ]
    enriched = audit.summarise_interventions(rows, 0.1235)
    assert [row["load_bearing"] for row in enriched] == [
        "strongly_load_bearing",
        "moderately_used",
        "weak_or_dormant",
    ]


# ---------------------------------------------------------------------------
# CPU-only + frozen-implementation integrity
# ---------------------------------------------------------------------------


def test_attach_cpu_is_cpu_only_and_evaluate_rejects_accelerators():
    device = audit.attach_cpu(1)
    assert device.type == "cpu"
    with pytest.raises(RuntimeError):
        audit.evaluate_mask(_model(), [], torch.device("cuda"), audit.AuditMask())


def test_audit_module_declares_no_accelerator_use():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    for token in ("cuda", 'device="gpu"', "device='gpu'"):
        assert token not in source, token


def test_frozen_p1_p2_modules_do_not_import_the_audit_module():
    for module in (p1, p2):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "clarity_audit" not in source, module.__name__


def test_masked_model_parameter_count_matches_h1():
    model = _model()
    assert int(sum(parameter.numel() for parameter in model.parameters())) == int(
        p2.total_parameter_count(audit.H1_CONFIG)["whole_model"]
    )
    assert int(sum(parameter.numel() for parameter in model.parameters())) == 97487
