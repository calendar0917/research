"""Focused wiring tests for zinc_cssd_nonlinear_binding_v1 (no full suite).

Only checks that can catch substantive errors: exact branch parameter
audits, the exact pooled-sum invariance of the separated branches under
pairing permutations (and responsiveness of the joint branches), endpoint
swap symmetry, fold grouping integrity and within-group permutation
semantics.  Stage-0-artifact-dependent model checks are skipped unless the
artifacts exist (they run after the stage-0 build, locally and on res-2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_cssd_nonlinear_binding_v1 as z
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_fresh_fold_replication_seed0_v1 as zfr,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw,
)

torch.set_num_threads(4)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed))


def test_branch_parameter_counts_are_exact():
    n_joint = z.NodeBranch("joint", _gen(1))
    n_sep = z.NodeBranch("sep", _gen(2))
    e_joint = z.EdgeBranch("joint", _gen(3))
    e_sep = z.EdgeBranch("sep", _gen(4))
    assert z._branch_parameter_count(n_joint) == z.N_JOINT_PARAMETERS == 7088
    assert z._branch_parameter_count(n_sep) == z.N_SEP_PARAMETERS == 7092
    assert z._branch_parameter_count(e_joint) == z.E_JOINT_PARAMETERS == 10208
    assert z._branch_parameter_count(e_sep) == z.E_SEP_PARAMETERS == 10240


def test_mlp_shapes_match_the_preregistered_table():
    assert [int(m.out_features) for m in z.NodeBranch("joint", _gen(1)).net if isinstance(m, torch.nn.Linear)] == [64, 48]
    sep = z.NodeBranch("sep", _gen(1))
    assert [int(m.out_features) for m in sep.net_s if isinstance(m, torch.nn.Linear)] == [44, 48]
    assert [int(m.out_features) for m in sep.net_c if isinstance(m, torch.nn.Linear)] == [44, 48]
    joint = z.EdgeBranch("joint", _gen(1))
    assert [int(m.out_features) for m in joint.net if isinstance(m, torch.nn.Linear)] == [64, 32]
    esep = z.EdgeBranch("sep", _gen(1))
    assert [int(m.out_features) for m in esep.net_s if isinstance(m, torch.nn.Linear)] == [53, 32]
    assert [int(m.out_features) for m in esep.net_c if isinstance(m, torch.nn.Linear)] == [53, 32]


def _toy_node_rows(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    zrows = torch.randn(n, 33, generator=g)
    qrows = torch.zeros(n, 28)
    qrows[torch.arange(n), torch.randint(0, 28, (n,), generator=g)] = 1.0
    return zrows, qrows


def test_node_sep_pooled_sum_is_pairing_invariant_joint_changes():
    zrows, qrows = _toy_node_rows(6, 11)
    joint = z.NodeBranch("joint", _gen(21)).eval()
    sep = z.NodeBranch("sep", _gen(22)).eval()
    with torch.no_grad():
        base_joint = joint(zrows, qrows).sum(0)
        base_sep = sep(zrows, qrows).sum(0)
        # permute the z<->q correspondence inside the group (multisets kept)
        perm = torch.as_tensor([1, 0, 3, 2, 5, 4])
        shuffled_joint = joint(zrows, qrows[perm]).sum(0)
        shuffled_sep = sep(zrows, qrows[perm]).sum(0)
    assert torch.allclose(base_sep, shuffled_sep, atol=1e-5), "N_sep pooled sum must not see the pairing"
    assert not torch.allclose(base_joint, shuffled_joint, atol=1e-5), "N_joint must respond to the pairing"


def _toy_edge_rows(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    z_u, z_w = torch.randn(n, 33, generator=g), torch.randn(n, 33, generator=g)
    q_u = torch.zeros(n, 28)
    q_w = torch.zeros(n, 28)
    q_u[torch.arange(n), torch.randint(0, 28, (n,), generator=g)] = 1.0
    q_w[torch.arange(n), torch.randint(0, 28, (n,), generator=g)] = 1.0
    b = torch.zeros(n, 4)
    b[:, 1] = 1.0
    return z_u, z_w, q_u, q_w, b


def test_edge_branches_are_endpoint_swap_symmetric():
    z_u, z_w, q_u, q_w, b = _toy_edge_rows(5, 13)
    joint = z.EdgeBranch("joint", _gen(31)).eval()
    sep = z.EdgeBranch("sep", _gen(32)).eval()
    with torch.no_grad():
        for branch in (joint, sep):
            base = branch(z_u, z_w, q_u, q_w, b)
            swapped = branch(z_w, z_u, q_w, q_u, b)
            assert torch.allclose(base, swapped, atol=1e-6)


def test_edge_sep_chemistry_tuple_perm_invariance_joint_changes():
    z_u, z_w, q_u, q_w, b = _toy_edge_rows(6, 17)
    joint = z.EdgeBranch("joint", _gen(41)).eval()
    sep = z.EdgeBranch("sep", _gen(42)).eval()
    with torch.no_grad():
        base_joint = joint(z_u, z_w, q_u, q_w, b).sum(0)
        base_sep = sep(z_u, z_w, q_u, q_w, b).sum(0)
        # permute complete chemistry endpoint tuples against the structure
        # endpoint pairs inside the group (same bond type)
        perm = torch.as_tensor([2, 3, 0, 1, 5, 4])
        shuf_joint = joint(z_u, z_w, q_u[perm], q_w[perm], b[perm]).sum(0)
        shuf_sep = sep(z_u, z_w, q_u[perm], q_w[perm], b[perm]).sum(0)
    assert torch.allclose(base_sep, shuf_sep, atol=1e-5), "E_sep pooled sum must not see the matching"
    assert not torch.allclose(base_joint, shuf_joint, atol=1e-5), "E_joint must respond to the matching"


def test_build_group_perm_only_reorders_within_groups():
    keys = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    perm = z.build_group_perm(keys, 7)
    assert sorted(perm.tolist()) == list(range(len(keys)))
    for key in np.unique(keys):
        assert sorted(perm[np.where(keys == key)[0]].tolist()) == sorted(np.where(keys == key)[0].tolist())


def test_group_keys_are_deterministic_and_disjoint_per_kind():
    class _Fake(torch.nn.Module):
        pass

    fake = _Fake()
    fake.env_occ_root = torch.tensor([0, 0, 1, 1])
    fake.env_occ_shell = torch.tensor([0, 1, 0, 2])
    fake.env_bond_root = torch.tensor([0, 1, 1])
    fake.env_bond_shellpair = torch.tensor([0, 3, 5])
    fake.env_bond_type = torch.tensor([1, 1, 2])
    assert z.n_group_keys(fake).tolist() == [0, 1, 3, 5]
    assert z.e_group_keys(fake).tolist() == [0 * 6 * 4 + 0 + 1, 1 * 24 + 3 * 4 + 1, 1 * 24 + 5 * 4 + 2]


@pytest.mark.skipif(not Path(zfr.TRAIN_LABEL_CSV).exists(), reason="train label csv not present")
def test_fold_smiles_groups_never_straddle():
    fold = z.build_fold()
    assert fold["checks"]["disjoint"] and fold["checks"]["cover"]
    moved_rows = sum(len(m["rows_moved_into_fit"]) for m in fold["moved"])
    assert fold["checks"]["sizes"]["fit"] == 8000 + moved_rows
    assert fold["checks"]["sizes"]["select"] + fold["checks"]["sizes"]["confirm"] == 2000 - moved_rows
    # no duplicate-SMILES group is split across two folds
    smi = pd.read_csv(zfr.TRAIN_LABEL_CSV)["smi_line"].to_numpy(np.int64)
    membership = {
        int(i): name
        for name, idx in (("fit", fold["fit_idx"]), ("select", fold["select_idx"]), ("confirm", fold["confirm_idx"]))
        for i in np.asarray(idx).tolist()
    }
    for value in set(smi.tolist()):
        rows = np.where(smi == value)[0].tolist()
        owners = {membership[int(r)] for r in rows}
        assert len(owners) == 1, f"smi group {value} is split across folds {owners}"


STAGE0_READY = (
    (z.RESULTS_DIR / "objects_manifest.json").exists()
    and (z.RESULTS_DIR / "cssd_basis.npz").exists()
    and (z.RESULTS_DIR / "Q_soup_state.pt").exists()
)


@pytest.mark.skipif(not STAGE0_READY, reason="stage-0 artifacts not built yet")
class TestArmsAgainstStage0:
    def test_build_all_arms_parameter_audit_and_shared_identity(self):
        objects = z.load_objects()
        basis = z.load_cssd_basis()
        payload = prev.TuplePayload(objects["payload_arrays"])
        kappa_M = float(objects["kappa"]["kappa_M"])
        models = {}
        hashes = {}
        for arm in z.ARMS:
            models[arm] = z.build_arm(arm, payload, kappa_M, basis, z.SEED)
            hashes[arm] = z.state_hash(z.shared_state(models[arm]))
            audit = z.parameter_audit(models[arm])
            assert audit["total_parameters"] == z._expected_arm_parameters(arm)
            assert audit["trainable_parameters"] == audit["total_parameters"]
        assert len(set(hashes.values())) == 1, "shared body init must be identical across arms"
        spread = max(
            z.parameter_audit(models[a])["total_parameters"] for a in ("C00", "C10", "C01", "C11")
        ) - min(
            z.parameter_audit(models[a])["total_parameters"] for a in ("C00", "C10", "C01", "C11")
        )
        assert spread <= 100

    def test_seed_one_is_genuinely_different(self):
        objects = z.load_objects()
        basis = z.load_cssd_basis()
        payload = prev.TuplePayload(objects["payload_arrays"])
        kappa_M = float(objects["kappa"]["kappa_M"])
        m0 = z.build_arm("C11", payload, kappa_M, basis, 0)
        m1 = z.build_arm("C11", payload, kappa_M, basis, 1)
        assert z.state_hash(z.shared_state(m0)) != z.state_hash(z.shared_state(m1))
        assert z.branch_seed_for("C11", 0) != z.branch_seed_for("C11", 1)
        _s0, h0 = zw.build_schedule(100, z.EPOCHS, 0 + z.TRAIN_SHUFFLE_OFFSET)
        _s1, h1 = zw.build_schedule(100, z.EPOCHS, 1 + z.TRAIN_SHUFFLE_OFFSET)
        assert h0 != h1
