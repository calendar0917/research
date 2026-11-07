"""CPU-only correctness tests for the LCRC-RIE frozen pilot.

No dataset, no checkpoint, no GPU: every test builds synthetic or hand-written
molecules and checks one frozen mechanism the round depends on.

* 8-family motif extraction, dedupe and family overlap,
* physical-graph distances (own BFS vs ``scipy.sparse.csgraph``),
* the frozen distance bins and the "drop short / drop cross-component" rules,
* SHUFFLE preserving motifs, pairs, pair features and per-bin counts,
* graph-level invariance (motif order, atom renumbering, pair endpoint swap),
* the ``sum / sqrt(1 + n)`` aggregation and the REAL-Train-only RMS scale,
* the 8,784-parameter head: zero-residual epoch 0, paired initialisation,
  save/reload equivalence,
* the frozen paired order plan and the pre-registered decision rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

import rie_common as C  # noqa: E402
import rie_features as F  # noqa: E402
import rie_motifs as MO  # noqa: E402

torch = pytest.importorskip("torch")
Chem = pytest.importorskip("rdkit.Chem")


# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------


def test_frozen_dimensions():
    assert C.PAIR_FEATURE_DIM == 165
    assert C.N_BINS == 3
    assert C.RELATION_DIM_FLAT == 495
    assert C.N_FAMILY_PAIRS == 36
    assert C.HEAD_PARAMETERS == 165 * 16 + 48 * 128 == 8784
    assert C.RELATION_SHAPE == (3, 165)


def test_family_pair_index_is_symmetric_and_bijective():
    seen = set()
    for a in range(C.N_MOTIF_FAMILIES):
        for b in range(C.N_MOTIF_FAMILIES):
            idx = C.family_pair_index(a, b)
            assert idx == C.family_pair_index(b, a)
            assert 0 <= idx < C.N_FAMILY_PAIRS
            seen.add(idx)
    assert len(seen) == C.N_FAMILY_PAIRS


def test_bin_for_distance_frozen_mapping():
    assert C.bin_for_distance(0) == -1
    assert C.bin_for_distance(2) == -1
    assert C.bin_for_distance(3) == 0
    assert C.bin_for_distance(4) == 0
    assert C.bin_for_distance(5) == 1
    assert C.bin_for_distance(6) == 1
    assert C.bin_for_distance(7) == 2
    assert C.bin_for_distance(25) == 2
    assert C.bin_for_distance(-1) == -1


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------


def _random_graph(rng, n, extra=2, disconnected_tail=0):
    edges = [(i, int(rng.integers(0, i))) for i in range(1, n - disconnected_tail)]
    for _ in range(extra):
        u, v = rng.integers(0, n - disconnected_tail, size=2)
        if u != v:
            edges.append((int(u), int(v)))
    return edges


def test_shortest_paths_matches_scipy_reference():
    rng = np.random.default_rng(0)
    for n in (5, 11, 17):
        for disconnected in (0, 3):
            edges = _random_graph(rng, n, disconnected_tail=disconnected)
            ours = F.shortest_paths(n, edges)
            ref = F.reference_shortest_paths(n, edges)
            assert ours.shape == (n, n)
            assert np.array_equal(ours, ref)
            if disconnected:
                assert (ours < 0).any()


def test_connected_components_and_cross_component_exclusion():
    n = 6
    edges = [(0, 1), (1, 2), (3, 4)]
    labels = F.connected_components(n, edges)
    assert labels[0] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert labels[5] != labels[0]
    dist = F.shortest_paths(n, edges)
    motifs = [np.asarray([0]), np.asarray([4])]
    assert F.motif_pair_distances(motifs, dist) == []
    assert F.eligible_pairs(motifs, dist) == []


# ---------------------------------------------------------------------------
# motifs
# ---------------------------------------------------------------------------


def test_motif_families_on_reference_molecules():
    motifs, reason = MO.motifs_for_smiles("CC(=O)N")
    assert reason is None
    families = {fam for fam, _ in motifs}
    assert C.FAMILY_NAME_TO_ID["Carbonyl"] in families
    assert C.FAMILY_NAME_TO_ID["Donor"] in families
    carbonyl = [atoms for fam, atoms in motifs if fam == C.FAMILY_NAME_TO_ID["Carbonyl"]]
    assert len(carbonyl) == 1 and len(carbonyl[0]) == 2

    halogens, _ = MO.motifs_for_smiles("FCBr")
    halogen = [atoms for fam, atoms in halogens if fam == C.FAMILY_NAME_TO_ID["Halogen"]]
    assert sorted(a for atoms in halogen for a in atoms) == [0, 2]  # F(0), Br(2)

    arom, _ = MO.motifs_for_smiles("c1ccccc1")
    aromatic = [atoms for fam, atoms in arom if fam == C.FAMILY_NAME_TO_ID["Aromatic"]]
    assert len(aromatic) == 1 and len(aromatic[0]) == 6

    # different families may share atoms: the carboxylic OH oxygen is both a
    # Donor and an Acceptor, and the carbonyl carbon belongs to two families
    shared, _ = MO.motifs_for_smiles("O=C(O)c1ccccc1")
    donor = [set(atoms) for fam, atoms in shared if fam == C.FAMILY_NAME_TO_ID["Donor"]]
    acceptor = [set(atoms) for fam, atoms in shared if fam == C.FAMILY_NAME_TO_ID["Acceptor"]]
    assert any(len(d & a) > 0 for d in donor for a in acceptor)
    carbonyl_atoms = [set(atoms) for fam, atoms in shared if fam == C.FAMILY_NAME_TO_ID["Carbonyl"]]
    assert carbonyl_atoms and carbonyl_atoms[0] == {0, 1}


def test_motif_atoms_recorded_as_sorted_unique_indices():
    motifs, _ = MO.motifs_for_smiles("OC(=O)c1ccccc1Cl")
    assert motifs == sorted(motifs, key=lambda item: (item[0], item[1]))
    for _fam, atoms in motifs:
        assert list(atoms) == sorted(set(atoms))
        assert len(atoms) >= 1


def test_parse_failure_is_reported_not_raised():
    motifs, reason = MO.motifs_for_smiles("this-is-not-a-smiles")
    assert motifs == []
    assert reason is not None


def test_molecule_structure_dedupes_duplicate_motif_matches():
    mol = Chem.MolFromSmiles("O=C(O)c1ccccc1")
    motifs = MO.motifs_for_mol(mol)
    doubled = list(motifs) + list(motifs)
    once = F.molecule_structure(n_atoms=mol.GetNumAtoms(), edges=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], motifs=motifs, official_row_id=11)
    twice = F.molecule_structure(n_atoms=mol.GetNumAtoms(), edges=[(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], motifs=doubled, official_row_id=11)
    assert once["n_motifs"] == twice["n_motifs"]
    assert np.array_equal(once["motif_family"], twice["motif_family"])
    assert np.array_equal(once["motif_atoms"], twice["motif_atoms"])


# ---------------------------------------------------------------------------
# SHUFFLE control
# ---------------------------------------------------------------------------


def test_shuffle_preserves_bin_multiset_and_is_deterministic():
    bins = [0, 0, 1, 2, 2, 2]
    first = F.shuffle_bins(bins, 12345)
    second = F.shuffle_bins(bins, 12345)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first), np.sort(np.asarray(bins)))
    others = {tuple(F.shuffle_bins(bins, row).tolist()) for row in range(200)}
    assert len(others) > 1


def test_shuffle_is_identity_when_all_pairs_share_one_bin():
    for bins in ([1, 1, 1, 1], [0], []):
        out = F.shuffle_bins(bins, 999)
        assert np.array_equal(out, np.asarray(bins, dtype=np.int8))


def test_shuffle_keeps_motifs_pairs_and_pair_features():
    """REAL and SHUFFLE must differ only in the bin assignment of each pair."""
    smiles = "O=C(O)CCC(=O)c1ccccc1Cl"
    mol = Chem.MolFromSmiles(smiles)
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    motifs = MO.motifs_for_mol(mol)
    struct = F.molecule_structure(n_atoms=mol.GetNumAtoms(), edges=edges, motifs=motifs, official_row_id=7)
    assert struct["n_pairs"] > 0
    assert np.array_equal(np.sort(struct["pair_bin"]), np.sort(struct["pair_bin_shuffle"]))
    assert np.bincount(struct["pair_bin"], minlength=3).tolist() == np.bincount(
        struct["pair_bin_shuffle"], minlength=3
    ).tolist()
    # the pair list itself is untouched by the shuffle
    assert struct["pair_a"].shape == struct["pair_b"].shape
    rng = np.random.default_rng(0)
    h = rng.standard_normal((mol.GetNumAtoms(), C.WIDTH)).astype(np.float32)
    out = F.molecule_relation_features(h, motifs, n_atoms=mol.GetNumAtoms(), edges=edges, official_row_id=7)
    assert np.array_equal(out["bin_counts_real"], out["bin_counts_shuffle"])


# ---------------------------------------------------------------------------
# relation features
# ---------------------------------------------------------------------------


def _two_motif_case(distance: int):
    """Two identical-family motifs separated by ``distance`` bonds on a chain."""
    n_atoms = distance + 1
    edges = [(i, i + 1) for i in range(n_atoms - 1)]
    motifs = [(0, (0,)), (0, (n_atoms - 1,))]
    return n_atoms, edges, motifs


def test_relation_features_are_zero_without_eligible_pairs():
    h = np.random.default_rng(1).standard_normal((4, C.WIDTH)).astype(np.float32)
    n_atoms, edges, motifs = _two_motif_case(distance=2)  # distance 2 -> dropped
    out = F.molecule_relation_features(h, motifs, n_atoms=n_atoms, edges=edges, official_row_id=5)
    assert out["real"].shape == C.RELATION_SHAPE
    assert np.abs(out["real"]).max() == 0.0
    assert np.abs(out["shuffle"]).max() == 0.0
    assert out["bin_counts_real"].tolist() == [0, 0, 0]


def test_relation_features_bin_assignment_for_known_distances():
    for distance, expected_bin in ((3, 0), (4, 0), (5, 1), (6, 1), (7, 2), (9, 2)):
        n_atoms, edges, motifs = _two_motif_case(distance)
        h = np.random.default_rng(distance).standard_normal((n_atoms, C.WIDTH)).astype(np.float32)
        out = F.molecule_relation_features(h, motifs, n_atoms=n_atoms, edges=edges, official_row_id=3)
        counts = out["bin_counts_real"].tolist()
        assert counts[expected_bin] == 1
        assert sum(counts) == 1


def test_relation_features_are_motif_order_invariant():
    mol = Chem.MolFromSmiles("O=C(O)CCC(=O)c1ccccc1Cl")
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    motifs = MO.motifs_for_mol(mol)
    h = np.random.default_rng(2).standard_normal((mol.GetNumAtoms(), C.WIDTH)).astype(np.float32)
    base = F.molecule_relation_features(h, motifs, n_atoms=mol.GetNumAtoms(), edges=edges, official_row_id=4)
    reversed_motifs = list(reversed(motifs))
    other = F.molecule_relation_features(
        h, reversed_motifs, n_atoms=mol.GetNumAtoms(), edges=edges, official_row_id=4
    )
    assert np.array_equal(base["real"], other["real"])
    assert np.array_equal(base["shuffle"], other["shuffle"])


def test_relation_features_are_atom_renumbering_invariant():
    mol = Chem.MolFromSmiles("O=C(O)CCC(=O)c1ccccc1Cl")
    n = mol.GetNumAtoms()
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    motifs = MO.motifs_for_mol(mol)
    rng = np.random.default_rng(3)
    h = rng.standard_normal((n, C.WIDTH)).astype(np.float32)
    base = F.molecule_relation_features(h, motifs, n_atoms=n, edges=edges, official_row_id=6)
    pi = rng.permutation(n)
    inv = np.empty_like(pi)
    inv[pi] = np.arange(n)
    edges_new = [(int(inv[u]), int(inv[v])) for u, v in edges]
    motifs_new = [(fam, tuple(sorted(int(inv[a]) for a in atoms))) for fam, atoms in motifs]
    h_new = np.empty_like(h)
    h_new[inv] = h
    other = F.molecule_relation_features(h_new, motifs_new, n_atoms=n, edges=edges_new, official_row_id=6)
    assert np.abs(base["real"] - other["real"]).max() <= 1e-5


def test_pair_features_are_symmetric_under_endpoint_swap():
    mol = Chem.MolFromSmiles("O=C(O)CCC(=O)c1ccccc1Cl")
    n = mol.GetNumAtoms()
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    motifs = MO.motifs_for_mol(mol)
    struct = F.molecule_structure(n_atoms=n, edges=edges, motifs=motifs, official_row_id=8)
    split = F.concat_structures([struct], rows=np.asarray([8], dtype=np.int64))
    gathered = F.gather_batch(split, np.asarray([0], dtype=np.int64), np.asarray([n], dtype=np.int64))
    h = np.random.default_rng(9).standard_normal((n, C.WIDTH)).astype(np.float32)
    z = F.motif_states(h, gathered)
    fwd = F.pair_features(z, gathered)
    swapped = dict(gathered)
    swapped["pair_a"], swapped["pair_b"] = gathered["pair_b"], gathered["pair_a"]
    assert np.array_equal(fwd, F.pair_features(z, swapped))


def test_aggregation_formula_matches_definition():
    # two pairs in key 0, one pair in key 3 -> sum/sqrt(1+n)
    features = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    keys = np.asarray([0, 0, 3], dtype=np.int64)
    out, counts = F.aggregate_features(features, keys, 6)
    expected0 = np.asarray([4.0, 6.0], dtype=np.float32) / np.sqrt(3.0)
    expected3 = np.asarray([5.0, 6.0], dtype=np.float32) / np.sqrt(2.0)
    assert counts.tolist() == [2, 0, 0, 1, 0, 0]
    assert np.allclose(out[0], expected0, atol=1e-6)
    assert np.allclose(out[3], expected3, atol=1e-6)
    assert np.all(out[[1, 2, 4, 5]] == 0.0)


# ---------------------------------------------------------------------------
# RMS normalisation
# ---------------------------------------------------------------------------


def test_rms_scale_is_train_only_rms_with_floor_and_keeps_zeros():
    sumsq = np.asarray([4.0, 0.0, 1e-12, 9.0], dtype=np.float64)
    scale, audit = F.rms_scale_from_sumsq(sumsq, 1)
    assert scale[0] == pytest.approx(2.0)
    assert scale[1] == pytest.approx(1.0)  # floored
    assert scale[2] == pytest.approx(1.0)  # below epsilon -> floored
    assert scale[3] == pytest.approx(3.0)
    assert audit["n_floored_columns"] == 2
    assert audit["fit_split"] == "real_train_only"
    block = np.zeros((2,) + C.RELATION_SHAPE, dtype=np.float32)
    scaled = F.apply_scale(block, np.ones(C.RELATION_DIM_FLAT, dtype=np.float32))
    assert np.abs(scaled).max() == 0.0


def test_feature_health_counts_zero_columns():
    block = np.zeros((4,) + C.RELATION_SHAPE, dtype=np.float32)
    block[:, 2, 0] = 1.0
    health = F.feature_health(block)
    assert health["n_zero_columns"] == C.RELATION_DIM_FLAT - 1
    assert health["nan_count"] == 0 and health["inf_count"] == 0


# ---------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------


def test_head_parameters_and_zero_residual_identity():
    import rie_model as RM

    head = RM.build_head(device="cpu")
    assert RM.head_parameter_count(head) == 8784
    assert set(head.state_dict()) == set(RM.head_parameter_names())
    assert torch.equal(head.task_weight.weight, torch.zeros_like(head.task_weight.weight))
    relation = torch.randn(7, C.N_BINS, C.PAIR_FEATURE_DIM)
    base = torch.randn(7, C.N_TASKS)
    with torch.no_grad():
        logits = base + head(relation)
        delta = head(relation)
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(logits, base)


def test_head_initial_state_is_paired_and_matches_default_linear_init():
    import rie_model as RM

    first = RM.initial_head_state(seed=0)
    second = RM.initial_head_state(seed=0)
    assert RM.state_hash(first) == RM.state_hash(second)
    assert all(torch.equal(first[k], second[k]) for k in first)
    torch.manual_seed(0)
    reference = torch.nn.Linear(C.PAIR_FEATURE_DIM, 16, bias=False)
    assert torch.equal(first["shared_relation_projection.weight"], reference.weight)
    # the private initialisation must not advance the global RNG stream
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    RM.initial_head_state(seed=0)
    assert torch.equal(before, torch.get_rng_state())


def test_head_save_reload_is_equivalent():
    import rie_model as RM

    head = RM.build_head(device="cpu")
    relation = torch.randn(5, C.N_BINS, C.PAIR_FEATURE_DIM)
    with torch.no_grad():
        before = head(relation)
    state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    reloaded = RM.build_head(state, device="cpu")
    with torch.no_grad():
        after = reloaded(relation)
    assert torch.equal(before, after)


def test_frozen_recipe_values():
    assert C.RECIPE["learning_rate"] == 3e-4
    assert C.RECIPE["weight_decay"] == 1e-3
    assert C.RECIPE["adam_betas"] == (0.9, 0.999)
    assert C.RECIPE["adam_eps"] == 1e-8
    assert C.RECIPE["gradient_clip_norm"] == 5.0
    assert C.RECIPE["batch_size"] == 1024
    assert C.RECIPE["epochs"] == 8
    assert C.RECIPE["eval_epochs"] == [0, 2, 4, 8]
    assert C.RECIPE["main_epoch"] == 8
    assert C.RECIPE["l1_coefficient"] == 1e-5
    assert C.RECIPE["dropout"] == 0.0
    assert C.RECIPE["scheduler"] == "none"
    assert C.RECIPE["early_stopping"] is False


def test_order_plan_is_deterministic_and_a_permutation():
    import rie_train as T

    plan_a = T.make_order_plan(seed=0, epochs=2, n_train=1000)
    plan_b = T.make_order_plan(seed=0, epochs=2, n_train=1000)
    assert T.plan_hash(plan_a) == T.plan_hash(plan_b)
    for rows in plan_a:
        assert np.array_equal(np.sort(rows), np.arange(1000))
    assert not np.array_equal(plan_a[0], np.arange(1000))
    assert T.plan_audit(plan_a)["is_permutation_per_epoch"] is True


# ---------------------------------------------------------------------------
# decision rule
# ---------------------------------------------------------------------------


def test_decision_rule_labels():
    import rie_compare as CMP

    label, reasons, checks = CMP.classify_decision(
        d_base=0.006, d_ctrl=0.004, d_ge20=0.005, improved_ge20=70, preconditions_ok=True, intervention_ok=True
    )
    assert label == "PROMISING_RELATION_SIGNAL" and reasons == [] and all(checks.values())

    label, reasons, checks = CMP.classify_decision(
        d_base=0.010, d_ctrl=0.0005, d_ge20=0.002, improved_ge20=40, preconditions_ok=True, intervention_ok=True
    )
    assert label == "WEAK_OR_MIXED_SIGNAL"
    assert "D_ctrl_ge_0.003" in reasons and "improved_ge20_tasks_ge_60" in reasons

    label, _, _ = CMP.classify_decision(
        d_base=0.001, d_ctrl=0.0005, d_ge20=0.0, improved_ge20=10, preconditions_ok=True, intervention_ok=True
    )
    assert label == "NO_RELATION_SIGNAL"

    label, reasons, _ = CMP.classify_decision(
        d_base=0.02, d_ctrl=0.01, d_ge20=0.01, improved_ge20=90, preconditions_ok=True, intervention_ok=False
    )
    assert label == "INSUFFICIENT_RELATION_INTERVENTION"

    label, _, _ = CMP.classify_decision(
        d_base=0.02, d_ctrl=0.01, d_ge20=0.01, improved_ge20=90, preconditions_ok=False, intervention_ok=True
    )
    assert label == "INCOMPLETE"
