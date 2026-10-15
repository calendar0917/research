"""Fast correctness tests for bjoc-mmp-feasibility-v1 (no ZINC data, no labels).

Covers the frozen single-cut extraction rules, direction/dedup/antisymmetry,
constitution and symmetry handling, canonicalisation invariance, exact
cross-context rule support, the scaffold-grouped leave-out unit, the frozen
zero fallback, and Phase 0A label isolation (the Phase 0A path must never be
able to call the label loader).
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_mmp_feasibility_v1 as mmp  # noqa: E402


def _mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _structures(n: int) -> list[dict]:
    return [{"row": i, "x": np.zeros(1), "edge_index": np.zeros((2, 0)), "edge_attr": np.zeros(0)} for i in range(n)]


def _regime(smiles_rows: list[str], spec: mmp.RegimeSpec | None = None) -> dict:
    smiles = list(smiles_rows)
    scaffolds = {i: (mmp.MurckoScaffold.MurckoScaffoldSmiles(mol=_mol(s)) or "") for i, s in enumerate(smiles)}
    core_members, _audit = mmp.extract_incidences(_structures(len(smiles)), smiles, excluded=())
    same_constitution = {i: mmp.constitution(_mol(s)) for i, s in enumerate(smiles)}
    return mmp.build_regime(
        core_members,
        spec or mmp.RegimeSpec("test", 1, None, None),
        same_constitution=same_constitution,
        scaffolds=scaffolds,
    )


# ---------------------------------------------------------------------------
# cut rule
# ---------------------------------------------------------------------------


def test_cut_bond_indices_excludes_ring_and_multiple_bonds():
    benzene = _mol("c1ccccc1")
    assert mmp.cut_bond_indices(benzene) == []
    ethylbenzene = _mol("CCc1ccccc1")
    indices = mmp.cut_bond_indices(ethylbenzene)
    # both acyclic single bonds are cuttable: the benzylic C-C and the
    # exocyclic C-ring bond (only the ring bonds themselves are excluded)
    assert len(indices) == 2
    for bond_idx in indices:
        bond = ethylbenzene.GetBondWithIdx(bond_idx)
        assert not bond.IsInRing()
        assert str(bond.GetBondType()) == "SINGLE"
    # the single C-C bond is cut; the acyclic double bond is not
    propene_indices = mmp.cut_bond_indices(_mol("CC=C"))
    assert len(propene_indices) == 1
    assert str(_mol("CC=C").GetBondWithIdx(propene_indices[0]).GetBondType()) == "SINGLE"


def test_fragment_cut_has_one_dummy_per_side_and_is_connected():
    mol = _mol("CCCO")
    for bond_idx in mmp.cut_bond_indices(mol):
        first, second = mmp.fragment_cut(mol, bond_idx)
        for piece in (first, second):
            dummies = [a for a in piece.GetAtoms() if a.GetAtomicNum() == 0]
            assert len(dummies) == 1
            assert piece.GetNumAtoms() >= 2  # dummy + at least one heavy atom


def test_no_pair_from_identical_molecules():
    regime = _regime(["CCO", "CCO"])
    assert regime["pairs"] == []


# ---------------------------------------------------------------------------
# direction, antisymmetry, dedup, multiple cutting schemes
# ---------------------------------------------------------------------------


def test_direction_is_fixed_by_fragment_keys_not_row_order():
    forward = _regime(["CCO", "CCN"])
    backward = _regime(["CCN", "CCO"])
    # CCO/CCN admit two single-cut granularities: (*C, *CN -> *CO) and
    # (*CC, *N -> *O); both are kept and both directions are normalised.
    assert len(forward["pairs"]) == 2 and len(backward["pairs"]) == 2
    fwd = {pair.core: pair for pair in forward["pairs"]}
    bwd = {pair.core: pair for pair in backward["pairs"]}
    assert set(fwd) == set(bwd) == {"*C", "*CC"}
    y = np.array([1.5, -0.5])
    for core, f in fwd.items():
        b = bwd[core]
        assert (f.r, f.rp) == (b.r, b.rp)
        assert f.r < f.rp and b.r < b.rp
        # the smaller fragment key is held by CCN (row 1 forward, row 0 backward)
        assert (f.row_r, f.row_rp) == (1, 0)
        assert (b.row_r, b.row_rp) == (0, 1)
        assert y[f.row_rp] - y[f.row_r] == pytest.approx(-(y[b.row_rp] - y[b.row_r]))


def test_multiple_cut_schemes_are_recorded_and_deduplicated():
    # isopropanol / isopropylamine: the two equivalent methyl cuts generate the
    # same (core *C, chain *C(C)O) incidence twice, so the molecule pair is
    # seen 4 times pre-dedup but kept once.
    regime = _regime(["CC(C)O", "CC(C)N"])
    assert len(regime["pairs"]) == 2
    for pair in regime["pairs"]:
        if pair.core == "*C":
            assert pair.pre_dedup_variants == 4
    assert regime["n_uncapped_pairs"] == 2
    assert regime["total_cut_realizations"] == 5  # 2*2 symmetric methyl cuts + one direct cut


def test_same_constitution_rows_do_not_form_an_edit():
    regime = _regime(["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"])
    assert regime["pairs"] == []


def test_canonicalisation_invariance_under_atom_relabelling():
    # 1) graph-level: renumbering atoms does not change the canonical SMILES
    mol = _mol("CCOCC")
    permutation = list(reversed(range(mol.GetNumAtoms())))
    assert Chem.MolToSmiles(Chem.RenumberAtoms(mol, permutation)) == Chem.MolToSmiles(mol)
    # 2) pipeline-level: equivalent non-canonical SMILES give identical pairs
    baseline = _regime(["CC(C)O", "CC(C)N"])
    relabelled = _regime(["OC(C)C", "NC(C)C"])
    keys_a = sorted((p.core, p.r, p.rp) for p in baseline["pairs"])
    keys_b = sorted((p.core, p.r, p.rp) for p in relabelled["pairs"])
    assert keys_a == keys_b and len(keys_a) > 0


# ---------------------------------------------------------------------------
# exact cross-context support
# ---------------------------------------------------------------------------


def test_repeated_rule_across_two_contexts_is_detected():
    # rule (*O -> *N) occurs with context *CC (ethanol/ethylamine) and with
    # context *C (methanol/methylamine)
    regime = _regime(["CCO", "CCN", "CO", "CN"])
    strict = regime["strict"]
    values = strict["support_values"]
    assert strict["n_repeated_rules"] >= 1
    assert int(values.max()) >= 2


def test_materialised_equals_uncapped_for_uncapped_regime():
    regime = _regime(["CCO", "CCN", "CCCO", "CCCN"])
    assert regime["n_uncapped_pairs"] == len(regime["pairs"])


# ---------------------------------------------------------------------------
# baselines / grouped evaluation
# ---------------------------------------------------------------------------


def _descriptor_stub(rows: int) -> dict[int, dict[str, np.ndarray]]:
    return {
        row: {
            "elements": np.zeros(len(mmp.ELEMENTS)),
            "ring_count": np.zeros(1),
            "aromatic_rings": np.zeros(1),
            "charge": np.zeros(1),
            "hbd": np.zeros(1),
            "hba": np.zeros(1),
            "rotatable": np.zeros(1),
        }
        for row in range(rows)
    }


def test_unknown_rule_falls_back_to_frozen_zero():
    pairs = {
        "row_r": [0, 1],
        "row_rp": [1, 2],
        "core": ["*C", "*C"],
        "r": ["*C", "*O"],
        "rp": ["*O", "*N"],
        "r_heavy": [1, 1],
        "rp_heavy": [1, 1],
        "scaffold_r": ["s1", "s2"],
        "scaffold_rp": ["s2", "s3"],
    }
    result = mmp.leave_scaffold_out_evaluation(pairs, np.array([1.0, -1.0]), _descriptor_stub(4))
    assert result["unknown_rule_predictions"] == 2  # every rule is unseen when its only group is held out
    assert result["baselines"]["rule_mean_fallback_zero"]["mae"] == pytest.approx(1.0)
    assert result["baselines"]["zero"]["mae"] == pytest.approx(1.0)


def test_group_unit_never_splits_a_scaffold_pair():
    pairs = {
        "scaffold_r": ["a", "a", "b"],
        "scaffold_rp": ["b", "b", "a"],
    }
    groups, unique = mmp._groups_by_scaffold_pair(pairs)
    assert groups[0] == groups[1] == groups[2] == "a|b"
    assert unique == ["a|b"]


# ---------------------------------------------------------------------------
# target formula signs
# ---------------------------------------------------------------------------


def test_formula_delta_signs_and_zero():
    constants = mmp.FORMULA_CONSTANTS_COMMUNITY
    same = mmp.formula_delta({"logP": 1.0, "sa": 1.0, "cycle": 0.0}, {"logP": 1.0, "sa": 1.0, "cycle": 0.0}, constants)
    assert same["d_formula"] == pytest.approx(0.0)
    up = mmp.formula_delta({"logP": 1.0, "sa": 1.0, "cycle": 0.0}, {"logP": 2.0, "sa": 1.0, "cycle": 0.0}, constants)
    assert up["d_formula"] > 0
    sa_up = mmp.formula_delta({"logP": 1.0, "sa": 1.0, "cycle": 0.0}, {"logP": 1.0, "sa": 3.0, "cycle": 0.0}, constants)
    assert sa_up["d_formula"] < 0
    cycle_up = mmp.formula_delta({"logP": 1.0, "sa": 1.0, "cycle": 0.0}, {"logP": 1.0, "sa": 1.0, "cycle": 3.0}, constants)
    assert cycle_up["d_formula"] < 0


def test_formula_y_uses_the_three_normalised_terms():
    constants = mmp.FORMULA_CONSTANTS_COMMUNITY
    comp = {"logP": constants["logP_mean"], "sa": -constants["SA_mean"], "cycle": -constants["cycle_mean"]}
    assert mmp.formula_y(comp, constants) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# label isolation
# ---------------------------------------------------------------------------


def test_phase0a_pure_path_never_calls_the_label_loader():
    with mock.patch.object(mmp, "load_train_labels", side_effect=AssertionError("labels touched in Phase 0A")):
        smiles = ["CCO", "CCN", "CCCO", "CCCN", "c1ccccc1O", "c1ccccc1N"]
        scaffolds = {i: mmp.MurckoScaffold.MurckoScaffoldSmiles(mol=_mol(s)) or "" for i, s in enumerate(smiles)}
        core_members, _ = mmp.extract_incidences(_structures(len(smiles)), smiles, excluded=())
        same_constitution = {i: mmp.constitution(_mol(s)) for i, s in enumerate(smiles)}
        spec = mmp.RegimeSpec("probe", 1, None, None)
        regime = mmp.build_regime(core_members, spec, same_constitution=same_constitution, scaffolds=scaffolds)
        mmp.regime_coverage(regime, spec, core_members=core_members, scaffolds=scaffolds)
