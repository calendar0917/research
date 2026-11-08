"""Relation statistics: definitions, exactness and the REAL/BLIND contrast."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import rc_checks as CHK
import rc_common as C
import rc_dist as D
import rc_relations as REL


# ---------------------------------------------------------------------------
# triangle numbering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7, 11, 16, 23, 40])
def test_triangle_index_matches_bruteforce(n):
    total = n * (n - 1) // 2
    p = torch.arange(total, dtype=torch.int64)
    nn = torch.full((total,), n, dtype=torch.int64)
    i, j = REL.triangle_index(p, nn)
    bi, bj = REL.brute_triangle_index(p, nn)
    assert torch.equal(i, bi)
    assert torch.equal(j, bj)


@pytest.mark.parametrize("n", [64, 200, 331, 332, 400])
def test_triangle_index_matches_bruteforce_large(n):
    total = n * (n - 1) // 2
    p = torch.arange(total, dtype=torch.int64)
    nn = torch.full((total,), n, dtype=torch.int64)
    i, j = REL.triangle_index(p, nn)
    bi, bj = REL.brute_triangle_index(p, nn)
    assert torch.equal(i, bi)
    assert torch.equal(j, bj)


def test_triangle_index_edge_cases():
    i, j = REL.triangle_index(torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.int64))
    assert i.numel() == 0 and j.numel() == 0
    # n = 1 has no pairs at all
    assert D.pair_counts_from_atoms(np.asarray([1]))[0] == 0


# ---------------------------------------------------------------------------
# bucket definitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("distance,code", [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 4), (7, 5), (30, 5)])
def test_distance_to_code_boundaries(distance, code):
    out = D.distance_to_code(np.asarray([[float(distance)]]))
    assert int(out[0, 0]) == code


def test_distance_to_code_rejects_bad_input():
    with pytest.raises(RuntimeError):
        D.distance_to_code(np.asarray([[0.0]]))
    with pytest.raises(RuntimeError):
        D.distance_to_code(np.asarray([[1.5]]))


def test_distance_to_code_marks_unreachable():
    out = D.distance_to_code(np.asarray([[np.inf, 3.0]]))
    assert int(out[0, 0]) == C.UNREACHABLE
    assert int(out[0, 1]) == 2


@pytest.mark.parametrize("name", sorted(CHK.TINY_GRAPHS))
def test_bucket_codes_are_exclusive_and_complete(name):
    n, edges = CHK.TINY_GRAPHS[name]
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    codes = D.molecule_bucket_codes(n, edges)
    ref = D.brute_force_distances(n, edges)
    iu = np.triu_indices(n, 1)
    assert codes.size == n * (n - 1) // 2
    assert np.array_equal(codes, D.distance_to_code(ref[iu]))
    # every reachable pair lands in exactly one bucket, the rest is the sentinel
    reachable = np.isfinite(ref[iu])
    assert np.array_equal(reachable, codes != C.UNREACHABLE)
    for index, (_label, lo, hi) in enumerate(C.BUCKETS):
        sel = reachable & (ref[iu] >= lo) & ((ref[iu] <= hi) if hi is not None else True)
        assert np.array_equal(sel, codes == index)


def test_disconnected_components_never_get_a_finite_distance():
    n, edges = CHK.TINY_GRAPHS["two_components"]
    codes = D.molecule_bucket_codes(n, np.asarray(edges, dtype=np.int64))
    # atoms {0,1,2} and {3,4} are separate components, atom 5 is isolated
    ref = D.brute_force_distances(n, np.asarray(edges, dtype=np.int64))
    iu = np.triu_indices(n, 1)
    d = ref[iu]
    cross = np.isinf(d)
    assert cross.any()
    assert np.all(codes[cross] == C.UNREACHABLE)
    assert np.all(codes[~cross] != C.UNREACHABLE)


def test_pair_counts_formula():
    n = np.asarray([1, 2, 3, 10], dtype=np.int64)
    assert D.pair_counts_from_atoms(n).tolist() == [0, 1, 3, 45]


# ---------------------------------------------------------------------------
# the statistics themselves
# ---------------------------------------------------------------------------


def _pairs(n: int, edges) -> list[tuple[int, int, int]]:
    """(u, v, bucket) for every unordered pair of one molecule."""
    iu = np.triu_indices(int(n), 1)
    codes = D.molecule_bucket_codes(int(n), np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    return [(int(u), int(v), int(b)) for u, v, b in zip(iu[0], iu[1], codes)]


def _roles(n_atoms: int, seed: int = 2) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn(n_atoms, C.N_ROLES, generator=gen, dtype=torch.float64)
    return torch.softmax(logits, dim=-1)


def test_statistics_match_explicit_enumeration():
    n, edges = CHK.TINY_GRAPHS["chain5"]
    pairs = _pairs(n, edges)
    R = _roles(n)
    for kind in ("real", "blind"):
        x, aux = REL.relation_input_from_roles(
            R, n_graphs=1, atom_counts=torch.tensor([n]), pairs=pairs, kind=kind
        )
        ref = CHK.brute_force_relation_input(R.numpy(), [n], pairs, kind)
        assert np.abs(x.numpy() - ref).max() < 1e-10


def test_t_d_is_symmetric_and_sums_to_144():
    n, edges = CHK.TINY_GRAPHS["ring6"]
    pairs = _pairs(n, edges)
    R = _roles(n)
    _x, aux = REL.relation_input_from_roles(
        R, n_graphs=1, atom_counts=torch.tensor([n]), pairs=pairs, kind="real"
    )
    T = aux["T"][0]
    for d in range(C.N_BUCKETS):
        assert torch.allclose(T[d], T[d].t(), atol=1e-12)
        if float(aux["ordered_counts"][0, d]) > 0:
            assert abs(float(T[d].sum()) - 144.0) < 1e-9
        else:
            assert float(T[d].abs().sum()) == 0.0


def test_empty_bucket_is_exactly_zero():
    n, edges = CHK.TINY_GRAPHS["chain5"]
    pairs = _pairs(n, edges)
    R = _roles(n)
    for kind in ("real", "blind"):
        _x, aux = REL.relation_input_from_roles(
            R, n_graphs=1, atom_counts=torch.tensor([n]), pairs=pairs, kind=kind
        )
        F = aux["F"][0]
        empties = [d for d in range(C.N_BUCKETS) if int(aux["ordered_counts"][0, d]) == 0]
        assert empties, "the 5-atom chain must have empty distant buckets"
        for d in empties:
            assert float(F[d].abs().sum()) == 0.0
        assert float(aux["w"][0, d]) == 0.0


def test_t_all_equals_weighted_sum_of_t_d():
    n, edges = CHK.TINY_GRAPHS["chain5"]
    pairs = _pairs(n, edges)
    R = _roles(n)
    _x, aux = REL.relation_input_from_roles(
        R, n_graphs=1, atom_counts=torch.tensor([n]), pairs=pairs, kind="real"
    )
    w = aux["w"][0].view(C.N_BUCKETS, 1, 1)
    recombined = (w * aux["T"][0]).sum(dim=0)
    assert torch.allclose(recombined, aux["T_all"][0], atol=1e-9)


def test_real_equals_blind_for_uniform_roles():
    result = CHK.check_uniform_roles_equivalence()
    assert result["passed"], result


def test_binding_change_affects_only_real():
    result = CHK.check_binding_change()
    assert result["passed"], result
    assert result["details"]["per_arm"]["blind"]["max_abs_diff"] <= 1e-7
    assert result["details"]["per_arm"]["real"]["max_abs_diff"] > 1e-3


# ---------------------------------------------------------------------------
# the geometry of a real batch
# ---------------------------------------------------------------------------


def test_pair_geometry_maps_to_same_molecule_pairs(tiny_batch):
    geom = REL.pair_geometry(tiny_batch)
    atom_batch = tiny_batch["atom_batch"]
    u, v = geom["u"], geom["v"]
    assert u.numel() == tiny_batch["pair_bucket"].numel()
    assert torch.all(u < v)
    # every pair is inside one molecule, and (u, v) is a distinct ordered key
    assert torch.equal(atom_batch[u], atom_batch[v])
    keys = torch.unique(u * 100000 + v)
    assert keys.numel() == u.numel()
    # per-molecule pair counts are exactly n(n-1)/2
    counts = tiny_batch["atom_counts"]
    assert torch.equal(tiny_batch["pair_counts"], counts * (counts - 1) // 2)
    # and the buckets agree with the batch's own bond graph
    rebuilt = CHK.rebuild_pairs_from_bonds(tiny_batch)
    assert np.array_equal(rebuilt[0], tiny_batch["pair_bucket"].numpy())
    assert np.array_equal(rebuilt[1], tiny_batch["pair_counts"].numpy())


def test_no_cross_molecule_cooccurrence(spec, norm, tiny_batch):
    result = CHK.check_no_cross_molecule(spec, norm, tiny_batch)
    assert result["passed"], result


def test_bruteforce_matches_batched_implementation(spec, norm, tiny_batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = CHK.check_cpu_vs_batched(spec, norm, tiny_batch, device=device)
    assert result["passed"], result
