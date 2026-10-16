"""Correctness tests for the exact attributed-graph canonicalisation.

Fresh-clone safe: synthetic graphs only, CPU, no data/, no checkpoints, no
torch.  Run with::

    uv run pytest -q tracks/structure-contribution/results/structure_context_support_v1/tests

The key property being tested is that ``canonical_form`` is an *exact* structural
identity: two attributed anchored graphs get the same key **iff** they are
isomorphic (no hash fingerprint is involved anywhere).
"""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from attributed_canon import canonical_form, validate_graph  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def random_graph(rng: random.Random, n: int, n_edges: int, n_attr: int = 3):
    attrs = [("C", rng.randrange(n_attr)) for _ in range(n)]
    pairs = list(itertools.combinations(range(n), 2))
    rng.shuffle(pairs)
    edges = [(u, v, ("SINGLE",)) for u, v in sorted(pairs[:n_edges])]
    return n, attrs, edges


def permute(n, attrs, edges, anchors, perm):
    """Relabel a graph by ``perm`` (new label of old vertex ``i`` is ``perm[i]``)."""
    new_attrs = [None] * n
    for i, a in enumerate(attrs):
        new_attrs[perm[i]] = a
    new_edges = []
    for u, v, ea in edges:
        a, b = perm[u], perm[v]
        new_edges.append((min(a, b), max(a, b), ea))
    return new_attrs, sorted(new_edges), [perm[a] for a in anchors]


def brute_isomorphic(n, attrs, edges, anchors, attrs2, edges2, anchors2) -> bool:
    """Exhaustive isomorphism test (small ``n`` only)."""
    for perm in itertools.permutations(range(n)):
        a2, e2, an2 = permute(n, attrs, edges, anchors, perm)
        if a2 == attrs2 and e2 == edges2 and sorted(an2) == sorted(anchors2):
            return True
    return False


# ---------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------


def test_relabelling_invariance_and_exactness_random():
    rng = random.Random(20260419)
    checked = 0
    for _ in range(120):
        n = rng.randrange(2, 7)
        n_edges = rng.randrange(0, min(n * (n - 1) // 2, 7) + 1)
        attrs, edges, anchors = None, None, None
        n, attrs, edges = random_graph(rng, n, n_edges, n_attr=2)
        anchors = rng.sample(range(n), 2)
        key = canonical_form(n, attrs, edges, anchors)
        perm = list(range(n))
        rng.shuffle(perm)
        a2, e2, an2 = permute(n, attrs, edges, anchors, perm)
        assert canonical_form(n, a2, e2, an2) == key, "relabelling changed the key"
        assert brute_isomorphic(n, attrs, edges, anchors, a2, e2, an2)
        checked += 1
    assert checked == 120


def test_certificate_separates_non_isomorphic_graphs():
    """No hash collisions: unequal structures must never share a key."""
    rng = random.Random(7)
    seen: dict[str, tuple] = {}
    collisions = 0
    for _ in range(400):
        n = rng.randrange(3, 7)
        n_edges = rng.randrange(1, 5)
        n, attrs, edges = random_graph(rng, n, n_edges, n_attr=2)
        anchors = rng.sample(range(n), 2)
        key = canonical_form(n, attrs, edges, anchors)
        canon = (n, attrs, edges, sorted(anchors))
        if key in seen:
            other = seen[key]
            if not brute_isomorphic(n, other[1], other[2], other[3], attrs, edges, anchors):
                collisions += 1
        else:
            seen[key] = canon
    assert collisions == 0
    assert len(seen) > 100


def test_attribute_sensitivity():
    """Element / H-count / bond order are all part of the patch identity."""
    edges = [(0, 1, ("SINGLE",)), (1, 2, ("SINGLE",))]

    def key(attrs, edges_=edges):
        return canonical_form(3, attrs, edges_, (0, 1))

    base = key([("C", ("H3",)), ("C", ("H2",)), ("O", ("H1",))])
    assert key([("C", ("H3",)), ("C", ("H2",)), ("N", ("H2",))]) != base          # element
    assert key([("C", ("H3",)), ("C", ("H1",)), ("O", ("H1",))]) != base          # H count
    assert key([("C", ("H3",)), ("C", ("H2",)), ("O", ("H1",))],
               [(0, 1, ("DOUBLE",)), (1, 2, ("SINGLE",))]) != base                # bond order


def test_anchors_are_an_unordered_set():
    n, attrs, edges = 4, [("C",), ("C",), ("C",), ("O",)], [(0, 1, ("SINGLE",)), (1, 2, ("SINGLE",)), (2, 3, ("SINGLE",))]
    assert canonical_form(n, attrs, edges, (0, 1)) == canonical_form(n, attrs, edges, (1, 0))
    assert canonical_form(n, attrs, edges, (0, 1)) != canonical_form(n, attrs, edges, (1, 2))


def test_anchors_are_part_of_the_identity():
    """Two isomorphic graphs with anchors moved are *not* the same anchored patch."""
    n, attrs, edges = 5, [("C",)] * 5, [(0, 1, ("SINGLE",)), (1, 2, ("SINGLE",)), (2, 3, ("SINGLE",)), (3, 4, ("SINGLE",))]
    assert canonical_form(n, attrs, edges, (0, 1)) != canonical_form(n, attrs, edges, (1, 2))


def test_return_order_is_a_consistent_vertex_order():
    """With a trivial automorphism group the vertex order is transported exactly."""
    rng = random.Random(11)
    for _ in range(30):
        n = rng.randrange(2, 7)
        n, attrs, edges = random_graph(rng, n, rng.randrange(0, 5), n_attr=2)
        # unique per-vertex attributes -> no non-trivial automorphism
        attrs = [a + (i,) for i, a in enumerate(attrs)]
        anchors = rng.sample(range(n), 2)
        cert, order = canonical_form(n, attrs, edges, anchors, return_order=True)
        assert sorted(order) == list(range(n))
        perm = list(range(n))
        rng.shuffle(perm)
        a2, e2, an2 = permute(n, attrs, edges, anchors, perm)
        cert2, order2 = canonical_form(n, a2, e2, an2, return_order=True)
        assert cert == cert2
        assert [perm[i] for i in order] == list(order2)


def test_empty_edges_and_single_vertex():
    assert canonical_form(1, [("C", ("H4",))], [], (0,)) == canonical_form(1, [("C", ("H4",))], [], (0,))
    assert canonical_form(1, [("C", ("H4",))], [], (0,)) != canonical_form(1, [("C", ("H3",))], [], (0,))
    two = canonical_form(2, [("C", ("H3",)), ("C", ("H3",))], [], (0, 1))
    assert two == canonical_form(2, [("C", ("H3",)), ("C", ("H3",))], [], (0, 1))


def test_validate_graph_rejects_malformed_inputs():
    attrs = [("C",)] * 3
    with pytest.raises(ValueError):
        validate_graph(3, attrs[:2], [], ())
    with pytest.raises(ValueError):
        validate_graph(3, attrs, [(0, 1, ()), (1, 0, ())], ())
    with pytest.raises(ValueError):
        validate_graph(3, attrs, [(0, 0, ())], ())
    with pytest.raises(ValueError):
        validate_graph(3, attrs, [(0, 5, ())], ())
    with pytest.raises(ValueError):
        validate_graph(3, attrs, [], (3,))
    validate_graph(3, attrs, [(0, 1, ()), (1, 2, ())], (0, 2))


# ---------------------------------------------------------------------------
# chemical sanity (needs RDKit; skipped if unavailable)
# ---------------------------------------------------------------------------


def _patch_keys(smiles: str, tier: str = "T1") -> list[str]:
    """S-key of every bond patch of a molecule, using the analysis pipeline."""
    pytest.importorskip("rdkit")
    import structure_context_support_v1 as scsv  # local import: keeps the graph tests torch-free

    return [inst.s_key for inst in scsv.instances_for_smiles(smiles, tier=tier)]


def _key_to_instance(smiles: str):
    pytest.importorskip("rdkit")
    import structure_context_support_v1 as scsv

    return {i.s_key: i for i in scsv.instances_for_smiles(smiles)}


def test_plain_aromatic_patch_recurs_across_benzene_toluene_pyridine_phenol():
    """The round's core condition at unit scale: one identical S, four contexts."""
    benzene = _key_to_instance("c1ccccc1")
    assert len(benzene) == 1, "all six benzene bonds are the same patch"
    for name in ("Cc1ccccc1", "c1ccncc1", "Oc1ccccc1"):
        other = _key_to_instance(name)
        shared = set(benzene) & set(other)
        assert shared, f"the plain aromatic CH-CH patch must recur in {name}"
        for key in shared:
            assert benzene[key].ct_primary != other[key].ct_primary, name
            assert benzene[key].ct_coarse != other[key].ct_coarse, name


def test_patch_identity_is_permutation_stable_on_a_real_molecule():
    pytest.importorskip("rdkit")
    import structure_context_support_v1 as scsv

    forward = _patch_keys("CC(=O)Nc1ccccc1")
    backward = _patch_keys("c1ccccc1NC(C)=O")
    assert sorted(forward) == sorted(backward)
    assert len(forward) == len(scsv.instances_for_smiles("CC(=O)Nc1ccccc1"))
