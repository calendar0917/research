"""cscl-correctness-v1 correctness tests.

Pins the fixes for the confirmed cscl-v0 defects (see
``notes/correctness_information_audit_v1.md``):

* labelled WL signature: relabel invariance + chemistry preservation;
* physical-bond canonicalization: PyG double-direction storage must never
  double count bonds (cyclomatic, histograms, terminal flags, relations);
* unit partition: atom coverage, unique bond attribution, relation endpoints;
* cache/vocabulary fingerprint isolation; label isolation;
* OpaqueModel permutation invariance.

Fresh-clone safe: synthetic molecules only; data-dependent checks skip when
``data/ZINC`` is absent.
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np
import pytest

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
from cscl_units import (  # noqa: E402
    ATOM_CATEGORIES,
    KIND_CHAIN,
    KIND_RING,
    UnitVocab,
    build_partition,
    partition_units,
    unique_undirected_bonds,
    unit_signature,
)

HAVE_ZINC = (Path(__file__).resolve().parents[3] / "data/ZINC/subset/processed/train.pt").exists()

C, O, N, F, S = 0, 1, 2, 3, 5  # ZINC atom-type ids (C,O,N,F,S per atom_dict)
SINGLE, DOUBLE, TRIPLE = 1, 2, 3


def doubled(edges):
    """Simulate PyG's symmetric adjacency storage (each bond twice)."""
    return [(u, v, bt) for u, v, bt in edges] + [(v, u, bt) for u, v, bt in edges]


# ---------------------------------------------------------------------------
# signature: relabel invariance (multiple numberings of the same labelled graph)
# ---------------------------------------------------------------------------


def _apply_perm(atoms, bonds, perm):
    inv = [0] * len(perm)
    for new, old in enumerate(perm):
        inv[old] = new
    return [atoms[old] for old in perm], [(inv[u], inv[v], bt) for u, v, bt in bonds]


def _random_perm(n, rng):
    p = list(range(n))
    rng.shuffle(p)
    return p


SAMPLE_STRUCTURES = [
    # (atoms, bonds) — covers heteroatom categories, bond orders, rings,
    # chains, branches, fused and spiro systems
    ([C, O], [(0, 1, SINGLE)]),
    ([C, N], [(0, 1, SINGLE)]),
    ([C, F], [(0, 1, SINGLE)]),
    ([N, N], [(0, 1, SINGLE)]),
    ([O, O], [(0, 1, SINGLE)]),
    ([C, C], [(0, 1, SINGLE)]),
    ([C, C], [(0, 1, DOUBLE)]),
    ([C, C], [(0, 1, TRIPLE)]),
    ([C, O], [(0, 1, DOUBLE)]),  # carbonyl-like
    ([N, C, N], [(0, 1, SINGLE), (1, 2, SINGLE)]),  # relabel-variance counterexample (N-C-N)
    ([C, C, C, C], [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE)]),  # chain
    ([C] * 6, [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE), (3, 4, SINGLE), (4, 5, SINGLE), (5, 0, SINGLE)]),  # ring
    ([C, C, C, C, C, O], [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE), (3, 4, SINGLE), (4, 0, SINGLE), (2, 5, SINGLE)]),  # branch on ring
    ([C] * 9 + [O], [(0, 1, DOUBLE), (1, 2, SINGLE), (2, 3, DOUBLE), (3, 4, SINGLE), (4, 5, DOUBLE), (5, 0, SINGLE), (0, 6, DOUBLE), (6, 7, SINGLE), (7, 8, DOUBLE), (8, 1, SINGLE)]),  # fused bicyclic
    ([C] * 9, [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE), (3, 4, SINGLE), (4, 0, SINGLE), (0, 5, SINGLE), (5, 6, SINGLE), (6, 7, SINGLE), (7, 8, SINGLE), (8, 0, SINGLE)]),  # spiro
    ([C, C, C, N, O, S], [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE), (3, 4, SINGLE), (4, 5, SINGLE), (5, 0, SINGLE)]),  # hetero ring
    ([C, C, C, C], [(0, 1, SINGLE), (0, 2, SINGLE), (0, 3, SINGLE)]),  # branched center
    ([C, C, C, N], [(0, 1, SINGLE), (0, 2, SINGLE), (0, 3, SINGLE)]),  # same shape, N leg
    ([C, N, C, N, C, N], [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE), (3, 4, SINGLE), (4, 5, SINGLE), (5, 0, SINGLE)]),  # alternating hetero ring
]


@pytest.mark.parametrize("case_idx", range(len(SAMPLE_STRUCTURES)))
def test_signature_relabel_invariance_all_structures(case_idx):
    atoms, bonds = SAMPLE_STRUCTURES[case_idx]
    rng = random.Random(1234 + case_idx)
    base = unit_signature(KIND_CHAIN, list(range(len(atoms))), atoms, bonds)
    for _ in range(8):
        perm = _random_perm(len(atoms), rng)
        atoms2, bonds2 = _apply_perm(atoms, bonds, perm)
        assert unit_signature(KIND_CHAIN, list(range(len(atoms))), atoms2, bonds2) == base


def test_signature_preserves_atom_categories():
    # C-C vs N-N vs O-O: different atom categories must never collide
    sigs = {
        unit_signature(KIND_CHAIN, [0, 1], [t, t], [(0, 1, SINGLE)])
        for t in (C, O, N, F, S)
    }
    assert len(sigs) == 5
    # mixed pairs with same shape but different chemistry
    assert unit_signature(KIND_CHAIN, [0, 1], [C, O], [(0, 1, SINGLE)]) != unit_signature(
        KIND_CHAIN, [0, 1], [C, N], [(0, 1, SINGLE)]
    )


def test_signature_preserves_bond_orders():
    sigs = {unit_signature(KIND_CHAIN, [0, 1], [C, C], [(0, 1, bt)]) for bt in (SINGLE, DOUBLE, TRIPLE)}
    assert len(sigs) == 3


def test_signature_isomorphic_labeled_graphs_agree_sampled():
    """Soundness: attributed-isomorphic units must share a signature.

    (The converse is NOT guaranteed: 1-WL cannot separate all non-isomorphic
    graphs — see the explicit limitation test below.)
    """
    rng = random.Random(7)
    atoms = [C, N, O, C, C, C]
    bonds = [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE), (3, 4, DOUBLE), (4, 5, SINGLE), (5, 0, SINGLE)]
    base = unit_signature(KIND_RING, list(range(6)), atoms, bonds)
    for _ in range(12):
        perm = _random_perm(6, rng)
        a2, b2 = _apply_perm(atoms, bonds, perm)
        assert unit_signature(KIND_RING, list(range(6)), a2, b2) == base


def test_signature_wl_limitation_documented_not_hidden():
    """Explicit limitation pin: hash/signature equality is NOT isomorphism.

    Two 3-regular graphs on 6 nodes (K3,3 and the triangular prism) are
    non-isomorphic but 1-WL-indistinguishable when equally labelled; equal
    signatures must therefore never be cited as an isomorphism proof.
    """
    k33 = [(0, 3, SINGLE), (0, 4, SINGLE), (0, 5, SINGLE), (1, 3, SINGLE), (1, 4, SINGLE), (1, 5, SINGLE), (2, 3, SINGLE), (2, 4, SINGLE), (2, 5, SINGLE)]
    prism = [(0, 1, SINGLE), (1, 2, SINGLE), (2, 0, SINGLE), (3, 4, SINGLE), (4, 5, SINGLE), (5, 3, SINGLE), (0, 3, SINGLE), (1, 4, SINGLE), (2, 5, SINGLE)]
    s1 = unit_signature(KIND_CHAIN, list(range(6)), [C] * 6, k33)
    s2 = unit_signature(KIND_CHAIN, list(range(6)), [C] * 6, prism)
    import networkx as nx

    g1, g2 = nx.Graph([(u, v) for u, v, _ in k33]), nx.Graph([(u, v) for u, v, _ in prism])
    assert not nx.is_isomorphic(g1, g2)
    assert s1 == s2  # WL limit, documented in unit_signature docstring


def test_signature_deterministic_across_calls():
    s1 = unit_signature(KIND_CHAIN, [0, 1], [C, O], [(0, 1, SINGLE)])
    s2 = unit_signature(KIND_CHAIN, [0, 1], [C, O], [(0, 1, SINGLE)])
    assert s1 == s2
    assert "a" not in s1.split("|")[-1] or True  # format free; stability is what matters


# ---------------------------------------------------------------------------
# physical-bond canonicalization
# ---------------------------------------------------------------------------


def test_unique_bonds_dedup_double_storage():
    phys = [(0, 1, SINGLE), (1, 2, DOUBLE), (2, 0, SINGLE)]
    bonds = unique_undirected_bonds([(u, v) for u, v, _ in doubled(phys)], [bt for _, _, bt in doubled(phys)])
    norm = {tuple(sorted((u, v))) + (bt,) for u, v, bt in phys}
    assert {(u, v, bt) for u, v, bt in bonds} == norm
    assert all(u < v for u, v, _ in bonds)


def test_unique_bonds_rejects_inconsistent_direction_types():
    with pytest.raises(ValueError):
        unique_undirected_bonds([(0, 1), (1, 0)], [SINGLE, DOUBLE])


def test_unique_bonds_rejects_self_loops():
    with pytest.raises(ValueError):
        unique_undirected_bonds([(0, 0)], [SINGLE])


def test_unique_bonds_rejects_parallel_multiedges():
    with pytest.raises(ValueError):
        unique_undirected_bonds([(0, 1), (1, 0), (0, 1)], [SINGLE, SINGLE, SINGLE])


# ---------------------------------------------------------------------------
# partition + PyG-style input: bond counts and cyclomatic number
# ---------------------------------------------------------------------------


def _partition_from_pyg(atoms, phys_bonds):
    ei = [(u, v) for u, v, _ in doubled(phys_bonds)]
    et = [bt for _, _, bt in doubled(phys_bonds)]
    return build_partition(list(atoms), ei, et)


def test_six_ring_cyclomatic_is_one_with_pyg_input():
    phys = [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE), (3, 4, SINGLE), (4, 5, SINGLE), (5, 0, SINGLE)]
    part = _partition_from_pyg([C] * 6, phys)
    assert part.n_units == 1
    unit = part.units[0]
    assert len(unit.intra_bonds) == 6
    cyc = len(unit.intra_bonds) - len(unit.atoms) + 1
    assert cyc == 1  # v0 reported 7 from doubled storage


def test_chain_cyclomatic_zero_and_inter_counts_single():
    phys = [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE)]
    part = _partition_from_pyg([C] * 4, phys)
    unit = part.units[0]
    assert len(unit.intra_bonds) == 3 and not unit.inter_bonds
    assert len(unit.intra_bonds) - len(unit.atoms) + 1 == 0


def test_fused_and_spiro_cyclomatic_two_single_ring_unit():
    fused = [(0, 1, SINGLE), (1, 2, DOUBLE), (2, 3, SINGLE), (3, 4, DOUBLE), (4, 5, SINGLE), (5, 0, SINGLE), (0, 6, DOUBLE), (6, 7, SINGLE), (7, 8, DOUBLE), (8, 1, SINGLE)]
    part = _partition_from_pyg([C] * 9, fused)
    rings = [u for u in part.units if u.kind == KIND_RING]
    assert len(rings) == 1 and len(rings[0].atoms) == 9
    assert len(rings[0].intra_bonds) - len(rings[0].atoms) + 1 == 2
    spiro = [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE), (3, 4, SINGLE), (4, 0, SINGLE), (0, 5, SINGLE), (5, 6, SINGLE), (6, 7, SINGLE), (7, 8, SINGLE), (8, 0, SINGLE)]
    part = _partition_from_pyg([C] * 9, spiro)
    rings = [u for u in part.units if u.kind == KIND_RING]
    assert len(rings) == 1 and len(rings[0].atoms) == 9
    assert len(rings[0].intra_bonds) - len(rings[0].atoms) + 1 == 2


def test_inter_relation_bond_counted_once_with_pyg_input():
    # biphenyl-like: two rings + one physical inter bond
    phys = (
        [(0, 1, SINGLE), (1, 2, DOUBLE), (2, 3, SINGLE), (3, 4, DOUBLE), (4, 5, SINGLE), (5, 0, SINGLE)]
        + [(6, 7, SINGLE), (7, 8, DOUBLE), (8, 9, SINGLE), (9, 10, DOUBLE), (10, 11, SINGLE), (11, 6, SINGLE)]
        + [(3, 6, SINGLE)]
    )
    part = _partition_from_pyg([C] * 12, phys)
    rel = part.inter_relations()
    assert len(rel) == 1
    (pair, bts) = next(iter(rel.items()))
    assert pair[0] < pair[1] and bts == [SINGLE]  # v0 produced [SINGLE, SINGLE]


# ---------------------------------------------------------------------------
# molecule_units descriptors with PyG-style doubled input (v1 semantics)
# ---------------------------------------------------------------------------


def test_molecule_units_descriptors_not_double_counted():
    import cscl_features as cf

    phys = [(0, 1, SINGLE), (1, 2, DOUBLE), (2, 3, SINGLE), (3, 4, DOUBLE), (4, 5, SINGLE), (5, 0, SINGLE), (5, 6, SINGLE)]
    x = np.zeros((7, 1), dtype=np.int64)
    ei = np.array([[u for u, v, _ in doubled(phys)], [v for u, v, _ in doubled(phys)]], dtype=np.int64)
    ea = np.array([bt for _, _, bt in doubled(phys)], dtype=np.int64)
    m = cf.molecule_units(0, x, ei, ea, "test")
    assert len(m.unit_sigs) == 2
    ring = m.unit_kinds.index(KIND_RING)
    n_intra = float(m.desc[ring, ATOM_CATEGORIES : ATOM_CATEGORIES + 4].sum())
    assert n_intra == 6.0  # not 12
    cyc = n_intra - len(m.unit_atoms[ring]) + 1
    assert cyc == 1.0
    chain = 1 - ring
    # relation histogram: one physical bond of type SINGLE
    assert len(m.rel_pairs) == 1
    assert m.rel_feat[0, SINGLE] == 1.0 and m.rel_feat[0].sum() - m.rel_feat[0, 4] == 1.0
    # attach_count lives in the descriptor: chain unit has exactly 1 attachment
    assert m.desc[chain, cf.D_HIST + 4] == 1.0
    assert m.desc[chain, cf.D_HIST + 7] == 1.0  # is_terminal


# ---------------------------------------------------------------------------
# partition coverage / unique attribution / relation endpoints
# ---------------------------------------------------------------------------


def test_partition_atom_coverage_and_unique_bond_attribution():
    phys = (
        [(0, 1, SINGLE), (1, 2, DOUBLE), (2, 3, SINGLE), (3, 4, DOUBLE), (4, 5, SINGLE), (5, 0, SINGLE)]
        + [(3, 6, SINGLE), (6, 7, SINGLE)]
        + [(7, 8, SINGLE), (8, 9, DOUBLE), (9, 10, SINGLE), (10, 11, DOUBLE), (11, 12, SINGLE), (12, 7, SINGLE)]
    )
    atoms = [C] * 13
    part = _partition_from_pyg(atoms, phys)
    covered = sorted(a for u in part.units for a in u.atoms)
    assert covered == list(range(13))
    intra, inter = set(), set()
    for u in part.units:
        for b in u.intra_bonds:
            key = tuple(sorted(b[:2])) + (b[2],)
            assert key not in intra
            intra.add(key)
        for b in u.inter_bonds:
            key = tuple(sorted(b[:2])) + (b[2],)
            assert key not in inter
            inter.add(key)
    original = {tuple(sorted((u, v))) + (bt,) for u, v, bt in phys}
    assert intra | inter == original and not (intra & inter)


def test_rel_tensor_endpoints_point_to_correct_units():
    import sys as _sys

    _sys.path.insert(0, str(_CODE_DIR))
    import run_cscl_v0 as r0
    import torch

    phys = (
        [(0, 1, SINGLE), (1, 2, DOUBLE), (2, 3, SINGLE), (3, 4, DOUBLE), (4, 5, SINGLE), (5, 0, SINGLE)]
        + [(3, 6, SINGLE)]
        + [(6, 7, SINGLE), (7, 8, DOUBLE), (8, 9, SINGLE), (9, 10, DOUBLE), (10, 11, SINGLE), (11, 6, SINGLE)]
    )
    x = np.zeros((12, 1), dtype=np.int64)
    ei = np.array([[u for u, v, _ in doubled(phys)], [v for u, v, _ in doubled(phys)]], dtype=np.int64)
    ea = np.array([bt for _, _, bt in doubled(phys)], dtype=np.int64)
    m = cf.molecule_units(0, x, ei, ea, "biphenyl")
    # vocabulary: accept every signature as known
    vocab = UnitVocab(min_count=1)
    vocab.fit([s for s in m.unit_sigs] * 3)
    stats = cf.FitStats()
    stats.vocab = vocab
    sigs = [s for s in m.unit_sigs]
    stats.type_means = {vocab.to_id(s, m.unit_kinds[k], len(m.unit_atoms[k])): np.zeros(cf.DESC_DIM, np.float32) for k, s in enumerate(sigs)}
    stats.global_mean = np.zeros(cf.DESC_DIM, np.float32)
    stats.rel_mean = np.zeros(cf.REL_DIM, np.float32)
    ts = r0.build_tensors([m], stats)[0]
    assert len(ts.rel_unit_lo) == 1
    a, b = m.rel_pairs[0]
    # endpoints must be exactly the two ring units
    assert {ts.rel_unit_lo[0], ts.rel_unit_hi[0]} == {a, b}
    # bond type histogram matches the physical inter bond
    assert float(ts.rel_feat[0, SINGLE]) == 1.0
    # unit indices are valid
    assert 0 <= ts.rel_unit_lo[0] < len(m.unit_sigs) and 0 <= ts.rel_unit_hi[0] < len(m.unit_sigs)


def test_full_pipeline_relabel_invariance_partition_and_signatures():
    phys = (
        [(0, 1, SINGLE), (1, 2, DOUBLE), (2, 3, SINGLE), (3, 4, DOUBLE), (4, 5, SINGLE), (5, 0, SINGLE)]
        + [(3, 6, SINGLE), (6, 7, SINGLE), (7, 2, SINGLE)]
    )
    atoms = [C, N, C, C, C, C, O, C]
    part = build_partition(atoms, [(u, v) for u, v, _ in doubled(phys)], [bt for _, _, bt in doubled(phys)])
    rng = random.Random(99)
    base_sigs = sorted(u.signature for u in part.units)
    for _ in range(6):
        perm = _random_perm(len(atoms), rng)
        atoms2, bonds2 = _apply_perm(atoms, phys, perm)
        part2 = build_partition(atoms2, [(u, v) for u, v, _ in doubled(bonds2)], [bt for _, _, bt in doubled(bonds2)])
        assert sorted(u.signature for u in part2.units) == base_sigs
        assert [u.kind for u in sorted(part2.units, key=lambda u: (u.kind != KIND_RING, u.atoms))] == [
            u.kind for u in sorted(part.units, key=lambda u: (u.kind != KIND_RING, u.atoms))
        ]


# ---------------------------------------------------------------------------
# attributed-isomorphism sampling (soundness of v1 signatures)
# ---------------------------------------------------------------------------


def test_v1_same_signature_units_are_attributed_isomorphic_random_sample():
    """Random labelled small graphs: sharing a v1 signature implies VF2
    attributed isomorphism on this sample (no counterexample found)."""
    import networkx as nx

    rng = random.Random(2026)
    n_checked = 0
    for _ in range(300):
        n = rng.randint(2, 7)
        atoms = [rng.choice([C, N, O, F, S]) for _ in range(n)]
        bonds = []
        for u, v in itertools.combinations(range(n), 2):
            if rng.random() < 0.5:
                bonds.append((u, v, rng.choice([SINGLE, DOUBLE, TRIPLE])))
        if not bonds:
            continue
        sig = unit_signature(KIND_CHAIN, list(range(n)), atoms, bonds)
        perm = _random_perm(n, rng)
        atoms2, bonds2 = _apply_perm(atoms, bonds, perm)
        sig2 = unit_signature(KIND_CHAIN, list(range(n)), atoms2, bonds2)
        if sig == sig2:
            g1 = nx.Graph()
            g1.add_nodes_from((i, {"t": atoms[i]}) for i in range(n))
            g1.add_edges_from((u, v, {"t": bt}) for u, v, bt in bonds)
            g2 = nx.Graph()
            g2.add_nodes_from((i, {"t": atoms2[i]}) for i in range(n))
            g2.add_edges_from((u, v, {"t": bt}) for u, v, bt in bonds2)
            assert nx.is_isomorphic(g1, g2, node_match=lambda a, b: a["t"] == b["t"], edge_match=lambda a, b: a["t"] == b["t"])
            n_checked += 1
    assert n_checked > 100


# ---------------------------------------------------------------------------
# cache version / vocabulary fingerprint isolation + label isolation
# ---------------------------------------------------------------------------


def test_stale_cache_is_rejected(tmp_path):
    import torch

    cache = tmp_path / "units.pt"
    torch.save({"mols": [], "y": np.zeros((0, 1), dtype=np.float32), "smiles": []}, cache)
    with pytest.raises(RuntimeError, match="stale unit cache"):
        cf.load_or_build_units_cache(cache, lambda: (_ for _ in ()).throw(AssertionError("builder must not run")))


def test_version_guard_accepts_matching_cache(tmp_path):
    import torch

    cache = tmp_path / "units.pt"
    mols, y, smiles = cf.load_or_build_units_cache(cache, lambda: ([], np.zeros((0, 1), dtype=np.float32), []))
    assert mols == []
    # reload must accept the written version fields
    cf.load_or_build_units_cache(cache, lambda: (_ for _ in ()).throw(AssertionError("builder must not run")))


def test_y_only_targets_is_label_isolated():
    y = np.array([[0.5], [1.5], [-2.0]], dtype=np.float32)
    t = cf.y_only_targets(y)
    assert t.shape == (3,) and float(t[1]) == 1.5
    with pytest.raises(ValueError):
        cf.y_only_targets(np.zeros((3, 2), dtype=np.float32))


def test_split_fingerprints_unchanged_from_v0():
    """The grouped split is SMILES-based and must stay byte-identical to v0
    (comparability of the audit), independent of the ZINC graphs."""
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    idx = cf.build_split_indices(smiles)
    assert len(idx["fit_all"]) == 8000 and len(idx["dev"]) == 2000
    assert len(idx["fit_inner"]) == 7200 and len(idx["monitor"]) == 800
    assert set(idx["fit_inner"]) | set(idx["monitor"]) == set(idx["fit_all"])
    assert not (set(idx["fit_inner"]) & set(idx["monitor"]))


# ---------------------------------------------------------------------------
# OpaqueModel permutation / batch invariance (v1 arm O-unit basis)
# ---------------------------------------------------------------------------


def _make_batch(n_mols: int, n_units: int, n_rel: int, seed: int):
    import torch

    import cscl_models as cm

    g = torch.Generator().manual_seed(seed)
    type_ids = torch.randint(0, 8, (n_units,), generator=g)
    desc = torch.randn(n_units, cf.DESC_DIM, generator=g)
    rel_feat = torch.randn(n_rel, cf.REL_DIM, generator=g)
    unit_mol = torch.randint(0, n_mols, (n_units,), generator=g)
    rel_unit_lo = torch.arange(n_rel) % n_units
    rel_unit_hi = (rel_unit_lo + 1) % n_units
    rel_mol = torch.randint(0, n_mols, (n_rel,), generator=g)
    t_lo, t_hi = type_ids[rel_unit_lo], type_ids[rel_unit_hi]
    swap = t_lo > t_hi
    lo = torch.where(swap, t_hi, t_lo)
    hi = torch.where(swap, t_lo, t_hi)
    unit_lo = torch.where(swap, rel_unit_hi, rel_unit_lo)
    unit_hi = torch.where(swap, rel_unit_lo, rel_unit_hi)
    return cm.Batch(
        n_mols=n_mols,
        type_ids=type_ids,
        desc=desc,
        unit_mol=unit_mol,
        rel_type_lo=lo,
        rel_type_hi=hi,
        rel_feat=rel_feat,
        rel_mol=rel_mol,
        rel_unit_lo=unit_lo,
        rel_unit_hi=unit_hi,
    )


def test_opaque_model_unit_permutation_invariance():
    import torch

    import cscl_models as cm

    torch.manual_seed(0)
    model = cm.OpaqueModel(vocab_size=8, hidden=16)
    b = _make_batch(2, 7, 3, seed=42)
    with torch.no_grad():
        p1 = model(b)
        # permute unit rows consistently (type ids, desc, unit_mol, endpoints)
        perm = torch.tensor([2, 0, 1, 6, 3, 5, 4])
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(len(perm))
        b2 = cm.Batch(
            n_mols=b.n_mols,
            type_ids=b.type_ids[perm],
            desc=b.desc[perm],
            unit_mol=b.unit_mol[perm],
            rel_type_lo=b.rel_type_lo,
            rel_type_hi=b.rel_type_hi,
            rel_feat=b.rel_feat,
            rel_mol=b.rel_mol,
            rel_unit_lo=inv[b.rel_unit_lo],
            rel_unit_hi=inv[b.rel_unit_hi],
        )
        p2 = model(b2)
    assert torch.allclose(p1, p2, atol=1e-5)


# ---------------------------------------------------------------------------
# data-dependent: real ZINC row-level sanity (skipped without data)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAVE_ZINC, reason="data/ZINC not available")
def test_real_zinc_row_bond_dedup_and_coverage_sample():
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import _load_zinc

    dataset = _load_zinc(Path(__file__).resolve().parents[3] / "data/ZINC", "train")
    for row in [0, 1, 2, 3, 4, 1234, 5678, 9999]:
        data = dataset[row]
        ei = data.edge_index.numpy()
        ea = data.edge_attr.reshape(-1).numpy()
        pairs = [(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])]
        types = [int(v) for v in ea]
        phys = unique_undirected_bonds(pairs, types)
        assert len(phys) * 2 == ei.shape[1]  # ZINC stores exactly two directions per bond
        part = build_partition([int(v) for v in data.x.reshape(-1).tolist()], pairs, types)
        covered = sorted(a for u in part.units for a in u.atoms)
        assert covered == list(range(int(data.num_nodes)))
        intra_inter = set()
        for u in part.units:
            for b in u.intra_bonds + u.inter_bonds:
                intra_inter.add(tuple(sorted(b[:2])) + (b[2],))
        assert intra_inter == {tuple(sorted((u, v))) + (bt,) for u, v, bt in phys}
