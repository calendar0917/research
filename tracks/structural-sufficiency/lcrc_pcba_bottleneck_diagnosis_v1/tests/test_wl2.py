"""WL2 unit tests: determinism, symmetry, Train-molecule frequency semantics,
label independence and row-order preservation on inputs that fail to parse."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import lc_common as C  # noqa: E402
import lc_structures as S  # noqa: E402


def colours(smi: str) -> list[bytes]:
    mol, _ = S.parse_smiles(smi)
    return S.wl2_root_colours(mol)


def test_colours_are_deterministic():
    a = colours("CC(=O)Oc1ccccc1C(=O)O")
    b = colours("CC(=O)Oc1ccccc1C(=O)O")
    assert a == b
    assert all(isinstance(x, bytes) and len(x) == 32 for x in a)


def test_benzene_symmetry_gives_one_environment():
    cs = colours("c1ccccc1")
    assert len(cs) == 6
    assert len(set(cs)) == 1


def test_radius_two_proxy_collides_on_para_carbon_of_pyridine():
    """A documented limitation of the frozen proxy, not a bug.

    The para carbon of pyridine has no heteroatom inside its 2-hop ball, so its
    WL2 colour equals a benzene carbon colour.  Everything else differs.  This is
    why the round calls WL2 a *support proxy*, never an isomorphism certificate.
    """
    benzene = set(colours("c1ccccc1"))
    pyridine = set(colours("c1ccncc1"))
    shared = benzene & pyridine
    assert len(benzene) == 1
    assert len(pyridine) == 4
    assert pyridine & benzene  # exactly the para-carbon collision
    assert len(shared) == 1


def test_bond_and_atom_attributes_enter_the_signature():
    """Same topology, different bond order -> different WL2 colours."""
    enol = set(colours("CC=CO"))
    keto_like = set(colours("CCC=O"))
    assert not (enol & keto_like)


def test_molecule_frequency_deduplicates_inside_one_molecule():
    support, meta = S.build_train_support(["c1ccccc1"], np.array([0], dtype=np.int64))
    assert meta["n_train_atoms"] == 6
    assert meta["n_distinct_wl2"] == 1
    assert list(support.values()) == [1]  # six identical atoms, one molecule


def test_molecule_frequency_counts_distinct_molecules():
    support, meta = S.build_train_support(
        ["c1ccccc1", "Cc1ccccc1"], np.array([0, 1], dtype=np.int64)
    )
    assert meta["n_train_atoms"] == 13
    # The plain aromatic carbon environment appears in both molecules.
    assert max(support.values()) == 2
    assert meta["n_distinct_wl2"] >= 3


def test_support_scope_is_train_only():
    support, _ = S.build_train_support(["c1ccccc1"], np.array([0], dtype=np.int64))
    rows, _ = S.support_features(["c1ccccc1", "c1ccncc1"], np.array([0, 1], dtype=np.int64), support, split="test")
    assert rows[0]["fraction_unseen"] == 0.0
    assert rows[0]["novelty_group"] == "SEEN"
    assert 0.0 < rows[1]["fraction_unseen"] <= 1.0
    assert rows[1]["novelty_group"] != "SEEN"


def test_feature_rows_keep_order_and_mark_unparsable_without_shifting():
    support, _ = S.build_train_support(["CCO"], np.array([0], dtype=np.int64))
    smiles = ["CCO", "definitely-not-a-smiles", "c1ccccc1"]
    rows, meta = S.support_features(smiles, np.array([0, 1, 2], dtype=np.int64), support, split="valid")
    assert [r["ogb_row"] for r in rows] == [0, 1, 2]
    assert [r["split_index"] for r in rows] == [0, 1, 2]
    assert rows[0]["novelty_group"] != "UNRESOLVED"
    assert rows[1]["novelty_group"] == "UNRESOLVED"
    assert rows[1]["parse_mode"] == "failed"
    assert rows[2]["novelty_group"] is not None
    assert meta["parse_modes"].get("failed", 0) == 1


def test_rare_frequency_uses_the_frozen_threshold():
    support, _ = S.build_train_support(
        ["CCO", "CCCO", "CCCCO", "OCCCCO", "CCCCCO"], np.array([0, 1, 2, 3, 4], dtype=np.int64)
    )
    rows, _ = S.support_features(["CCO"], np.array([0], dtype=np.int64), support, split="valid")
    frac_rare = rows[0]["fraction_rare"]
    per_atom = [support[c] for c in colours("CCO")]
    expected = float(np.mean([v <= C.RARE_SUPPORT_MAX for v in per_atom]))
    assert frac_rare == pytest.approx(expected)


def test_structure_module_never_touches_activity_labels():
    source = (CODE / "lc_structures.py").read_text(encoding="utf-8")
    for forbidden in ("graph-label", "task.labels", "split_labels", "y_test", "y_valid"):
        assert forbidden not in source


def test_shared_novelty_thresholds_are_single_constants():
    assert S.C.NOVELTY_PARTIAL_MAX == C.NOVELTY_PARTIAL_MAX
    assert S.C.RARE_SUPPORT_MAX == C.RARE_SUPPORT_MAX
    # The split-specific CSVs (if already built) must use the same labels.
    for split in ("valid", "test"):
        path = C.ANALYSIS / f"structure_support_{split}.csv"
        if not path.is_file():
            continue
        import pandas as pd

        df = pd.read_csv(path)
        assert set(df["novelty_group"].unique()) <= set(C.NOVELTY_GROUPS)
        recomputed = df["fraction_unseen"].map(lambda x: C.novelty_group(float(x) if np.isfinite(float(x)) else None))
        assert np.array_equal(recomputed.to_numpy(), df["novelty_group"].to_numpy())
