"""CPU tests for the branch construction and the two-level group statistics."""

from __future__ import annotations

import numpy as np
import torch

import rhic_audit as A
import rhic_branches as B
import rhic_common as C


def _small_batch(task, n_atoms, edges):
    return A.synthetic_batch(task.spec, n_atoms, edges, seed=7)


def test_branch_ids_match_independent_enumeration(task):
    cases = A.small_graph_audits(task.spec, _dummy_model(task))
    assert cases["all_ok"]
    for name, case in cases["cases"].items():
        assert case["branch_ids_match_independent_enumeration"], name
        assert case["branch_membership_matches"], name
        assert case["root_branch_counts_match"], name
        assert case["q_zero_for_n_lt_2"], name
        assert case["cross_zero_for_d_lt_2"], name


def _dummy_model(task):
    import rhic_model as M

    return M.build_rhic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")[0]


def test_branch_ids_use_the_storage_direction(task):
    batch = _small_batch(task, 4, [(0, 1), (1, 2), (1, 3)])
    bid = B.branch_ids(
        batch["cpc_path_atom"], batch["cpc_middle"], batch["cpc_bond_ij"], batch["bond_index"], verify=True
    )
    # path (0,1,2): physical bond 0 stored as (0,1) -> 0->1 -> id 0
    assert int(bid[0]) == 0
    flip = dict(batch)
    flip["bond_index"] = batch["bond_index"][:, [1, 0]].contiguous()
    bid_flip = B.branch_ids(
        batch["cpc_path_atom"], batch["cpc_middle"], batch["cpc_bond_ij"], flip["bond_index"], verify=True
    )
    # the same physical bond now reads v->u, so its branch label becomes 2e+1
    assert int(bid_flip[0]) == 1
    # the grouping is unchanged: paths sharing (root, middle) still share a label
    same_groups = (bid == bid[0]) == (bid_flip == bid_flip[0])
    assert bool(same_groups.all())


def test_group_layout_compacts_only_active_branches():
    bid = torch.tensor([0, 0, 3, 3, 3, 5], dtype=torch.long)
    counts, active, inv, active_ids = B.branch_group_layout(bid, 8)
    assert counts.tolist() == [2, 0, 0, 3, 0, 1, 0, 0]
    assert active.tolist() == [True, False, False, True, False, True, False, False]
    assert active_ids.tolist() == [0, 3, 5]
    assert inv.tolist() == [0, 0, 1, 1, 1, 2]
    assert torch.equal(counts.index_select(0, active_ids), torch.bincount(inv))


def test_stable_pair_term_equals_the_naive_formula():
    g = torch.Generator().manual_seed(3)
    v = torch.randn((37, 64), generator=g)
    inv = torch.tensor([0] * 2 + [1] * 1 + [2] * 5 + [3] * 7 + [4] * 22, dtype=torch.long)
    counts = torch.bincount(inv, minlength=5)
    _sums, mu, disp = B.group_mean_dispersion(v, inv, counts)
    q = B.pairwise_cooccurrence(mu, disp, counts)
    ref = B.reference_two_level_stats(v.double().numpy(), inv.tolist(), n_groups=5)
    mu_model = mu.double().numpy()[inv.numpy()]
    q_model = q.double().numpy()[inv.numpy()]
    assert np.abs(mu_model - ref["mu"][ref["inv"]]).max() < 1e-6
    assert np.abs(q_model - ref["q"][ref["inv"]]).max() < 1e-6
    # identical tokens: q == v*v, and a single-element group is exactly zero
    same = torch.ones((4, 3))
    c = torch.tensor([2, 2], dtype=torch.long)
    _s, m2, d2 = B.group_mean_dispersion(same, torch.tensor([0, 0, 1, 1]), c)
    q2 = B.pairwise_cooccurrence(m2, d2, c)
    assert torch.allclose(q2, torch.ones((2, 3)), atol=1e-7)
    assert int((counts < 2).sum()) == 1
    assert torch.equal(q[counts < 2], torch.zeros_like(q[counts < 2]))


def test_pair_term_excludes_self_products():
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    counts = torch.tensor([2])
    _s, mu, disp = B.group_mean_dispersion(x, torch.tensor([0, 0]), counts)
    q = B.pairwise_cooccurrence(mu, disp, counts)
    # distinct pairs: x0*x1 once = [[0,0],[0,0]] -> q == 0, whereas the self
    # product mean would be ((1,0)+(0,4))/2 = (0.5, 2)
    assert torch.allclose(q, torch.zeros_like(q), atol=1e-7)


def test_reference_root_level_matches_explicit_pair_sums():
    bv = np.asarray([[1.0, -2.0], [0.5, 0.25], [3.0, 1.0]], dtype=np.float64)
    out = B.reference_root_level(bv, [0, 0, 1], 2)
    # root 0 has two branches, so q is the single distinct-pair product
    expected = np.asarray([[1.0 * 0.5, -2.0 * 0.25]])
    assert np.allclose(out["q"][0:1], expected)
    assert np.allclose(out["q"][1:2], 0.0)
    assert np.allclose(out["counts"], [2, 1])


def test_per_atom_structure_matches_brute_force(task_with_cpc):
    cache = task_with_cpc.cpc
    ptr = np.asarray(cache["cpc_ptr"])
    mid = np.asarray(cache["cpc_middle"])
    got = B.per_atom_branch_structure(cache)
    pc = np.diff(ptr)
    bc = np.zeros_like(pc)
    mb = np.zeros_like(pc)
    mp = np.zeros_like(pc)
    for a in range(pc.size):
        ms = mid[ptr[a] : ptr[a + 1]]
        if ms.size == 0:
            continue
        runs, cur = [], 1
        for i in range(1, ms.size):
            if ms[i] == ms[i - 1]:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        bc[a] = len(runs)
        mb[a] = max(runs)
        mp[a] = sum(1 for r in runs if r >= 2)
    assert np.array_equal(got["path_count"], pc)
    assert np.array_equal(got["branch_count"], bc)
    assert np.array_equal(got["max_paths_in_branch"], mb)
    assert np.array_equal(got["multi_path_branches"], mp)


def test_batch_local_offsets_and_isolation(task_with_cpc):
    task = task_with_cpc
    rows = [int(task.train_rows[0]), int(task.train_rows[1]), int(task.valid_rows[0])]
    out = A.batch_local_branch_audit(task, task.cpc, rows)
    assert out["all_ok"], out["checks"]
    assert out["n_active_branches"] > 0


def test_multi_molecule_isolation_exact_on_cpu(task_with_cpc):
    import rhic_model as M

    task = task_with_cpc
    model, _ = M.build_rhic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    model.eval()
    rows = [int(x) for x in task.train_rows[:4]]
    whole = C.P.collate_with_cpc(task, task.cpc, rows)
    left = C.P.collate_with_cpc(task, task.cpc, rows[:2])
    right = C.P.collate_with_cpc(task, task.cpc, rows[2:])
    with torch.inference_mode():
        a = model(whole)
        b = model(left)
        c = model(right)
    # GEMM blocking can change the last bits with the batch size; the semantics
    # (no path crosses a molecule boundary) must hold to fp32 tolerance.
    assert torch.allclose(a[:2], b, atol=1e-5, rtol=0.0)
    assert torch.allclose(a[2:], c, atol=1e-5, rtol=0.0)
