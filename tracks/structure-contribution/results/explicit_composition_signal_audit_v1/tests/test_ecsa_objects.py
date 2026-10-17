"""Correctness tests for the explicit-composition signal audit (label-free).

These tests use small fixed molecules; they do not need the ZINC tensors and do
not read any target.  The full-data integration proof is the Stage-A run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import ecsa_objects as obj  # noqa: E402


FIXTURES = (
    "CCO",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)(C)c1ccc(O)cc1",
    "C1CCCCC1",
    "O=C(Nc1ccccc1)c1ccccc1",
)

#: a real ZINC pair with identical B0 marginals but different constitution
#: (one of the 11 Stage-A strict marginal collisions; no label is used here)
COLLISION_PAIR = (
    "COC[C@H](O)CNC(=O)N[C@H]1CCC[C@@H]1c1ccc(F)cc1",
    "C[C@H](CNC(=O)N[C@@H]1CCC[C@@H]1CO)Oc1ccccc1F",
)

#: a pair used in the frozen Stage-A output; kept local so the test is stable
COLLISION_PAIR_2 = (
    "Cc1cccc([N+](=O)[O-])c1S(=O)(=O)Oc1cccc(C(C)C)c1",
    "Cc1c([N+](=O)[O-])cccc1S(=O)(=O)Oc1cccc(C(C)C)c1",
)


def _record(smiles: str, row: int = 0, uid: int = 0):
    from rdkit import Chem

    from structure_context_support_v1 import build_mol_rec

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    Chem.RemoveStereochemistry(mol)
    rec = build_mol_rec("test", row, uid, smiles, mol)
    return obj.make_record(row, uid, smiles, mol, rec)


def _renumbered_records(smiles: str, trials: int = 4):
    from rdkit import Chem

    from structure_context_support_v1 import build_mol_rec

    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    Chem.RemoveStereochemistry(mol)
    base = obj.make_record(0, 0, smiles, mol, build_mol_rec("test", 0, 0, smiles, mol))
    rng = np.random.default_rng(7)
    out = [base]
    for _ in range(trials):
        perm = list(range(mol.GetNumAtoms()))
        rng.shuffle(perm)
        new_mol = Chem.RenumberAtoms(mol, [int(p) for p in perm])
        new_rec = obj.make_record(0, 0, smiles, new_mol, build_mol_rec("test", 0, 0, smiles, new_mol))
        out.append(new_rec)
    return out


def test_b0_ethanol_marginals():
    rec = _record("CCO")
    assert rec.n_heavy == 3
    assert rec.n_bonds == 2
    assert sum(c for _, c in rec.atom_class_counts) == 3
    assert sum(c for _, c in rec.bond_class_counts) == 2
    assert rec.line_graph_degrees == (1, 1)
    assert rec.junction_counts == (0, 0)
    assert rec.n_junctions == 0


def test_b0_benzene_marginals():
    rec = _record("c1ccccc1")
    assert rec.n_heavy == 6
    assert rec.n_bonds == 6
    assert rec.line_graph_degrees == (2, 2, 2, 2, 2, 2)
    assert rec.junction_counts == (0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("smiles", FIXTURES)
def test_b0_b1_c_are_invariant_under_atom_renumbering(smiles):
    records = _renumbered_records(smiles)
    first = records[0]
    for other in records[1:]:
        assert other.b0_key == first.b0_key
        assert other.b1_key == first.b1_key
        assert other.b1_t2_key == first.b1_t2_key
        assert other.b1_t3_key == first.b1_t3_key
        assert other.c_key_hex == first.c_key_hex
        assert other.c_smiles == first.c_smiles


@pytest.mark.parametrize("smiles", FIXTURES)
def test_patch_union_reconstructs_all_bonds(smiles):
    rec = _record(smiles)
    assert rec.reconstruction_ok is True


def test_c_key_is_exact_on_small_graphs():
    key1 = obj.colored_graph_key(
        ["a", "a", "b"], [(0, 1), (1, 2)]
    )
    key2 = obj.colored_graph_key(
        ["b", "a", "a"], [(2, 1), (1, 0)]
    )
    key3 = obj.colored_graph_key(
        ["a", "a", "a"], [(0, 1), (1, 2)]
    )
    key4 = obj.colored_graph_key(
        ["a", "a", "a"], [(0, 1), (1, 2), (2, 0)]
    )
    assert key1 == key2
    assert key1 != key3
    assert key3 != key4


def test_strict_marginal_collision_is_a_real_counterexample():
    a = _record(COLLISION_PAIR[0], row=1)
    b = _record(COLLISION_PAIR[1], row=2)
    assert a.b0_key == b.b0_key
    assert a.c_smiles != b.c_smiles
    assert a.b1_key != b.b1_key
    assert a.c_key_hex != b.c_key_hex
    # second real collision, positional isomer on the same scaffold
    x = _record(COLLISION_PAIR_2[0], row=3)
    y = _record(COLLISION_PAIR_2[1], row=4)
    assert x.b0_key == y.b0_key
    assert x.c_smiles != y.c_smiles
    assert x.scaffold == y.scaffold


def test_same_constitution_has_same_c_even_for_different_smiles_order():
    from rdkit import Chem

    smi = "CC(=O)Oc1ccccc1C(=O)O"
    mol = Chem.MolFromSmiles(smi)
    alt = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    a = _record(smi, row=1)
    b = _record(alt, row=2)
    assert a.c_smiles == b.c_smiles
    assert a.c_key_hex == b.c_key_hex
    assert a.b0_key == b.b0_key
    assert a.b1_key == b.b1_key


def test_tier_m0_and_m01_separate_the_real_collision():
    from collections import defaultdict

    records = [_record(COLLISION_PAIR[0], row=1), _record(COLLISION_PAIR[1], row=2)]
    m0_groups = obj.group_records(records, obj.tier_key("M0"))
    m01_groups = obj.group_records(records, obj.tier_key("M01"))
    m0 = obj.summarize_groups(records, m0_groups, "M0", obj.TIER_DEFINITIONS["M0"])
    m01 = obj.summarize_groups(records, m01_groups, "M01", obj.TIER_DEFINITIONS["M01"])
    assert m0["n_diffC_groups"] == 1
    assert m0["n_pairs_diffC"] == 1
    assert m01["n_diffC_groups"] == 0
    assert m01["n_pairs_diffC"] == 0


def test_pair_count_closed_form_does_not_count_same_C_pairs():
    from collections import defaultdict

    records = [
        _record("CCO", row=1),
        _record("CCO", row=2),  # duplicate constitution
        _record("COC", row=3),  # constitutional isomer
    ]
    # force a shared B0 group for the test by using a constant key
    groups = {("synthetic",): [0, 1, 2]}
    summary = obj.summarize_groups(records, groups, "synthetic", "test")
    # CCO/CCO same C, CCO/COC different C: only 2 cross-C pairs, not 3
    assert summary["n_pairs_diffC"] == 2
    # A mixed group is a diffC group; its same-C pair is not double-counted.
    assert summary["sameC_duplicate_groups"] == []


def test_label_loader_in_structure_module_never_touches_y():
    import torch

    from structure_context_support_v1 import load_pyg_structure_only

    class ExplodingY:
        def __getitem__(self, key):
            raise AssertionError("y was accessed by the structure loader")

    blob = (
        {
            "x": torch.tensor([[0]], dtype=torch.long),
            "edge_index": torch.empty((2, 0), dtype=torch.long),
            "edge_attr": torch.empty((0,), dtype=torch.long),
            "y": ExplodingY(),
        },
        {"x": [0, 1], "edge_index": [0, 0]},
        None,
    )
    records = load_pyg_structure_only([0], blob=blob)
    assert len(records) == 1
    assert set(records[0]) == {"row", "x", "edge_index", "edge_attr"}
    assert not hasattr(obj, "load_official_train_y")


def test_cluster_bootstrap_is_deterministic_and_uses_clusters():
    values = [1.0, 2.0, 3.0, 4.0]
    clusters = ["a", "a", "b", "b"]
    out1 = obj.cluster_bootstrap_mean(values, clusters, n_boot=200, seed=123)
    out2 = obj.cluster_bootstrap_mean(values, clusters, n_boot=200, seed=123)
    assert out1["n_clusters"] == 2
    assert out1["mean"] == pytest.approx(2.5)
    assert out1["ci95"] == out2["ci95"]
    assert out1["ci95"][0] <= out1["mean"] <= out1["ci95"][1]


def test_r2_and_ols_shapes():
    y = [0.0, 1.0, 2.0, 3.0]
    design = [[0.0], [1.0], [2.0], [3.0]]
    out = obj.ols_with_intercept(y, design)
    assert out["r2"] == pytest.approx(1.0)
    assert out["rank"] == 2
    assert obj.r2_score([0.0, 1.0], [0.5, 0.5]) == pytest.approx(0.0)
