"""Correctness tests for ``structure_context_support_v1``.

Fresh-clone safe: synthetic inputs, CPU, no ``data/``, no checkpoints, no
training, no labels.  Run with::

    uv run pytest -q tracks/structure-contribution/results/structure_context_support_v1/tests

What is enforced here rather than asserted:

* the ZINC structure loader can never reach a label tensor (a fake ``y`` that
  raises on any access is injected into the collated blob);
* the label-free Lipophilicity objects are rejected if they carry ``y``;
* the closed-form matched-pair counting equals brute-force pair enumeration;
* the pre-registered decision rule maps statistics to verdicts as frozen in
  ``../PROTOCOL.md`` §6.
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

import structure_context_support_v1 as scsv  # noqa: E402


# ---------------------------------------------------------------------------
# label isolation
# ---------------------------------------------------------------------------


class _Bomb:
    """Label sentinel: any read/iteration/formatting raises."""

    def _boom(self, *args, **kwargs):
        raise AssertionError("the label tensor was accessed")

    __getattr__ = _boom
    __getitem__ = _boom
    __iter__ = _boom
    __len__ = _boom
    __repr__ = _boom
    __str__ = _boom
    __eq__ = _boom
    __ne__ = _boom


def _fake_pyg_blob(n_atoms: int = 3, n_rows: int = 2):
    """Structural look-alike of the official PyG collated blob, plus a fake label."""
    torch = pytest.importorskip("torch")

    x = torch.arange(n_atoms * n_rows, dtype=torch.long).reshape(-1, 1) % 28
    edge_index = []
    edge_attr = []
    for r in range(n_rows):
        edge_index += [[0, 1], [1, 0], [1, 2], [2, 1]]
        edge_attr += [1, 1, 2, 2]
    data = {
        "x": x,
        "edge_index": torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.long),
        "y": _Bomb(),
    }
    slices = {
        "x": torch.arange(0, (n_rows + 1) * n_atoms, n_atoms, dtype=torch.long),
        "edge_index": torch.arange(0, (n_rows + 1) * 4, 4, dtype=torch.long),
    }
    return (data, slices, None)


def test_pyg_loader_never_reads_labels():
    blob = _fake_pyg_blob()
    recs = scsv.load_pyg_structure_only([0, 1], blob=blob)
    assert len(recs) == 2
    for rec in recs:
        assert set(rec) == {"row", "x", "edge_index", "edge_attr"}
        assert rec["x"].shape == (3,)
        assert rec["edge_index"].shape == (4, 2)
        assert rec["edge_attr"].shape == (4,)
        assert not any(isinstance(v, _Bomb) for v in rec.values())


def test_label_free_guard_rejects_objects_with_y():
    from types import SimpleNamespace

    scsv.assert_label_free([SimpleNamespace(smiles="CCO"), SimpleNamespace(smiles="CCN")])
    with pytest.raises(AssertionError):
        scsv.assert_label_free([SimpleNamespace(smiles="CCO"), SimpleNamespace(smiles="CCN", y=1.0)])


# ---------------------------------------------------------------------------
# closed-form pair counting
# ---------------------------------------------------------------------------


def test_matched_discordant_pairs_matches_brute_force():
    rng = random.Random(5)
    for _ in range(60):
        n_inst = rng.randrange(2, 9)
        entries = []
        for _ in range(n_inst):
            entries.append(
                (
                    rng.choice([10, 10, 11, 12]),           # n_heavy
                    rng.choice(["C5N1", "C5N1", "C6N1"]),   # formula
                    rng.choice([0, 1]),                     # rings
                    rng.choice(["a", "b", "c"]),            # context
                    1,
                )
            )
        # expand to an instance list and brute force the pair statistics
        flat = []
        for nh, f, r, ct, c in entries:
            flat += [(nh, f, r, ct)] * c
        for key_fn, name in (
            (lambda nh, f, r: (nh,), "size"),
            (lambda nh, f, r: (nh, f), "size_formula"),
            (lambda nh, f, r: (nh, f, r), "exact_stratum"),
            (lambda nh, f, r: (nh // 2, scsv._strip_carbon(f), r), "near_stratum"),
        ):
            grouped = {}
            for nh, f, r, ct in flat:
                grouped.setdefault(key_fn(nh, f, r), {}).setdefault(ct, 0)
                grouped[key_fn(nh, f, r)][ct] += 1
            brute = 0
            for per_ct in grouped.values():
                brute += sum(
                    ca * cb
                    for (cta, ca), (ctb, cb) in itertools.combinations(per_ct.items(), 2)
                    if cta != ctb
                )
            assert scsv._matched_discordant_pairs(entries, key_fn) == brute, name


def test_strip_carbon():
    assert scsv._strip_carbon("C12Cl2N2O3S1") == "Cl2N2O3S1"
    assert scsv._strip_carbon("C7") == ""
    assert scsv._strip_carbon("N2O1") == "N2O1"


# ---------------------------------------------------------------------------
# frozen decision rule
# ---------------------------------------------------------------------------


def _decision(n_types=100, c2=50, hetero=20, ladder_exact=10):
    return {
        "n_types_mol5_scaf2": n_types,
        "n_types_mol5_scaf2_two_contexts_2mol": c2,
        "n_chemically_distinctive_types": hetero,
        "n_types_with_two_contexts_in_one_stratum": {
            "size": 20,
            "size_formula": 10,
            "exact_stratum": ladder_exact,
            "near_stratum": 15,
        },
    }


def test_criteria_thresholds_are_frozen():
    t = scsv.DECISION_THRESHOLDS
    assert t["C1_min_types_mol_support_5_scaffold_2"] == 10
    assert t["C2_min_types_two_contexts_each_2_molecules"] == 5
    assert t["C3_min_chemically_distinctive_types"] == 3
    assert t["C4_min_types_with_2_contexts_in_one_exact_stratum"] == 5
    assert scsv.STRATUM_LEVELS == ("size", "size_formula", "exact_stratum", "near_stratum")


def test_verdict_mapping():
    good = scsv.evaluate_criteria(_decision())
    assert all(good.values())
    # a single failing criterion is enough to move the verdict
    assert not scsv.evaluate_criteria(_decision(c2=1))["C2_independent_context_diversity"]
    assert not scsv.evaluate_criteria(_decision(ladder_exact=0))["C4_non_trivial_context_variation"]
    assert not scsv.evaluate_criteria(_decision(n_types=1))["C1_support_breadth"]
    assert not scsv.evaluate_criteria(_decision(hetero=0))["C3_chemical_distinctiveness"]

    assert scsv.decide({"a": _decision()}, True)["verdict"] == "CONTEXT_SUPPORT_PRESENT"
    assert (
        scsv.decide({"a": _decision(ladder_exact=0)}, True)["verdict"]
        == "ONLY_TRIVIAL_CONTEXT_VARIATION"
    )
    # one dataset has the full core (C1&C2&C3) but loses it under C4, the other
    # has no independent contexts at all -> the failure mode of the *core* dataset wins
    assert (
        scsv.decide({"a": _decision(c2=0), "b": _decision(ladder_exact=0)}, True)["verdict"]
        == "ONLY_TRIVIAL_CONTEXT_VARIATION"
    )
    # no dataset keeps the core (C1&C2&C3) -> insufficient support
    assert (
        scsv.decide({"a": _decision(c2=0), "b": _decision(n_types=1)}, True)["verdict"]
        == "INSUFFICIENT_CROSS_GRAPH_SUPPORT"
    )
    assert (
        scsv.decide({"a": _decision(n_types=0), "b": _decision(hetero=0, ladder_exact=0)}, True)["verdict"]
        == "INSUFFICIENT_CROSS_GRAPH_SUPPORT"
    )
    # one dataset passing all four lifts the combined verdict, even if the other fails
    assert (
        scsv.decide({"zinc": _decision(ladder_exact=0), "lipo": _decision()}, True)["verdict"]
        == "CONTEXT_SUPPORT_PRESENT"
    )
    assert scsv.decide({"a": _decision()}, False)["verdict"] == "INCONCLUSIVE"


# ---------------------------------------------------------------------------
# structural plumbing (RDKit only, no data/)
# ---------------------------------------------------------------------------


def test_instances_per_bond_and_patch_identity():
    pytest.importorskip("rdkit")
    insts = scsv.instances_for_smiles("CC(=O)N")
    assert len(insts) == 3  # heavy-atom bonds only
    assert all(i.u != i.v for i in insts)
    assert all(i.s_key and i.ct_primary and i.ct_coarse for i in insts)
    # the three bonds share one patch graph but have different anchor identities
    assert len({i.s_key for i in insts}) == 3


def test_same_patch_different_outer_shell_unit_scale():
    """The core data condition, at unit scale: same S, different radius-2 context."""
    pytest.importorskip("rdkit")
    acid = {i.s_key: i for i in scsv.instances_for_smiles("CCC(=O)O")}
    phenyl = {i.s_key: i for i in scsv.instances_for_smiles("O=C(O)Cc1ccccc1")}
    common = set(acid) & set(phenyl)
    assert common, "the carboxylic acid patches must be identical in both molecules"
    for key in common:
        assert acid[key].ct_primary != phenyl[key].ct_primary
        assert acid[key].ct_coarse != phenyl[key].ct_coarse


def test_scaffold_and_formula_fields():
    pytest.importorskip("rdkit")
    from rdkit import Chem

    mol = Chem.MolFromSmiles("Oc1ccccc1")
    rec = scsv.build_mol_rec("test", 0, 0, "Oc1ccccc1", mol)
    assert rec.heavy_formula == "C6O1"
    assert rec.n_rings == 1
    assert rec.scaffold == "c1ccccc1"
    assert rec.n_heavy == 7
