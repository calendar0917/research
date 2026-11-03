"""Unit and frozen-artefact tests for the LCRC failure-mechanism P0 diagnostic."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import p0_common as C  # noqa: E402
import p0_sources as SRC  # noqa: E402
import p0_stats as STAT  # noqa: E402
import p0_structure as STR  # noqa: E402


# ---------------------------------------------------------------------------
# small SMILES fixture
# ---------------------------------------------------------------------------

FIXTURE_SMILES = [
    "CCO",
    "CCN",
    "c1ccccc1",
    "CC(=O)O",
    "CCC",
    "CCOC",
    "CCCl",
    "CCBr",
    "c1ccncc1",
    "CC(C)O",
    "C[N+](C)(C)C",
    "O=C([O-])c1ccccc1",
]


@pytest.fixture(scope="module")
def fixture_molecules():
    return STR.build_molecule_table(FIXTURE_SMILES)


@pytest.fixture(scope="module")
def frozen_group_a():
    return SRC.load_group_a()


@pytest.fixture(scope="module")
def frozen_group_b():
    return SRC.load_group_b()


# ---------------------------------------------------------------------------
# metric arithmetic and direction
# ---------------------------------------------------------------------------


def test_hand_computed_sse_rmse_mae():
    y = np.asarray([1.0, 2.0, 3.0])
    pred = np.asarray([1.0, 4.0, 1.0])
    m = STAT.metrics(y, pred)
    assert m["sse"] == pytest.approx(8.0, abs=1e-15)
    assert m["mse"] == pytest.approx(8.0 / 3.0, abs=1e-15)
    assert m["rmse"] == pytest.approx((8.0 / 3.0) ** 0.5, abs=1e-15)
    assert m["mae"] == pytest.approx(4.0 / 3.0, abs=1e-15)
    assert m["p90_abs"] == pytest.approx(2.0, abs=1e-15)


def test_d_i_sign_is_pna_minus_lcrc_squared_error():
    y = np.asarray([1.0, 2.0, 3.0])
    lcrc = np.asarray([4.0, 4.0, 4.0])
    pna = np.asarray([1.0, 2.0, 3.0])
    d = STAT.squared_error(y, lcrc) - STAT.squared_error(y, pna)
    assert np.all(d > 0)  # PNA is perfect, so positive means PNA better
    assert np.array_equal(d, np.asarray([9.0, 4.0, 1.0]))


def test_top_k_count_and_tail_definition():
    assert STAT.top_k_count(420) == 42
    assert STAT.top_k_count(1) == 1
    err = np.asarray([0.1, 5.0, 0.2, 4.9, 0.0])
    idx = STAT.top_error_indices(err, 0.4)  # k = 2
    assert sorted(idx.tolist()) == [1, 3]


def test_quartile_bins_are_fixed_and_monotone():
    values = np.asarray([0.0] * 8 + [1.0, 2.0, 3.0, 4.0])
    labels, edges = STAT.quartile_bins(values, 4)
    assert labels.shape == values.shape
    assert np.all(np.diff(edges) >= 0)
    # values equal to the first edge are placed in the higher bin index
    assert labels[values == 0.0].min() >= 1


# ---------------------------------------------------------------------------
# strict alignment
# ---------------------------------------------------------------------------


def test_strict_align_rejects_shuffled_file_and_row_id_recovers():
    rows = np.asarray([10, 20, 30, 40], dtype=np.int64)
    pred = np.asarray([1.0, 2.0, 3.0, 4.0])
    order = np.asarray([3, 0, 2, 1])
    with pytest.raises(SRC.AlignmentError):
        SRC.strict_align(rows[order], pred[order], rows, label="test")
    recovered = SRC.align_by_row_id(rows[order], pred[order], rows, label="test")
    assert np.array_equal(recovered, pred)


def test_align_rejects_duplicate_row_ids():
    rows = np.asarray([1, 1, 2], dtype=np.int64)
    with pytest.raises(SRC.AlignmentError):
        SRC.align_by_row_id(
            rows, np.asarray([1.0, 2.0, 3.0]), np.asarray([1, 2], dtype=np.int64), label="dup"
        )


def test_row_shuffle_does_not_change_metrics():
    rng = np.random.default_rng(0)
    y = rng.normal(size=50)
    pred = y + rng.normal(size=50)
    rows = np.arange(50, dtype=np.int64)
    base = STAT.metrics(y, pred)
    order = rng.permutation(50)
    shuffled = STAT.metrics(
        y,
        SRC.align_by_row_id(rows[order], pred[order], rows, label="shuffle"),
    )
    assert base == shuffled


# ---------------------------------------------------------------------------
# rooted radius-2 signature
# ---------------------------------------------------------------------------


def test_signature_renumbering_invariance(fixture_molecules):
    audit = STR.permutation_invariance_audit(fixture_molecules, n_molecules=12, seed=3)
    assert audit["renumbering_mismatches"] == 0
    assert audit["passed"] is True


def test_signature_encodes_root_position(fixture_molecules):
    mol = Chem.MolFromSmiles("CC(C)O")
    signatures, _ = STR.molecule_signatures(mol)
    # atoms 0 and 2 are the two equivalent methyl carbons; atom 1 is the
    # central carbon.  A root-position-blind signature would make all three equal.
    assert signatures[1] != signatures[0]
    assert signatures[0] == signatures[2]


def test_signature_label_codes_include_bond_order_charge_and_ring_flag():
    mol = Chem.MolFromSmiles("CC")
    ring_info = mol.GetRingInfo()
    single = STR._bond_label_code(mol.GetBondWithIdx(0))
    rw = Chem.RWMol(mol)
    rw.GetBondWithIdx(0).SetBondType(Chem.BondType.DOUBLE)
    double = STR._bond_label_code(rw.GetBondWithIdx(0))
    assert single != double

    neutral = STR._atom_label_code(mol, 0, ring_info)
    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(0).SetFormalCharge(1)
    charged_mol = rw.GetMol()
    charged_mol.UpdatePropertyCache(strict=False)
    charged = STR._atom_label_code(charged_mol, 0, charged_mol.GetRingInfo())
    assert neutral != charged

    ring_mol = Chem.MolFromSmiles("C1CCCCC1")
    in_ring = STR._atom_label_code(ring_mol, 0, ring_mol.GetRingInfo())
    chain_mol = Chem.MolFromSmiles("CCCCCC")
    not_in_ring = STR._atom_label_code(chain_mol, 0, chain_mol.GetRingInfo())
    assert in_ring != not_in_ring


def test_signature_is_self_consistent_under_renumbering_per_atom():
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
    base, _ = STR.molecule_signatures(mol)
    rng = np.random.default_rng(7)
    for _ in range(5):
        perm = list(range(mol.GetNumAtoms()))
        rng.shuffle(perm)
        permuted = Chem.RenumberAtoms(mol, [int(x) for x in perm])
        sigs, _ = STR.molecule_signatures(permuted)
        for new_idx, old_idx in enumerate(perm):
            assert sigs[new_idx] == base[old_idx]


def test_signature_collision_audit_small(fixture_molecules):
    audit = STR.signature_collision_audit(fixture_molecules, max_molecules=12, max_pairs=50, seed=1)
    # only same-signature pairs are expected to be isomorphic; any different
    # signature pair that is isomorphic would be a collision and fails the test.
    assert audit["different_signature_isomorphic_pairs"] == 0


# ---------------------------------------------------------------------------
# support dictionary
# ---------------------------------------------------------------------------


def test_document_frequency_counts_documents_not_atoms():
    sigs = [["A", "A", "B"], ["A", "C"], ["B", "B"]]
    df = STR.document_frequency(sigs)
    assert df == {"A": 2, "B": 2, "C": 1}


def test_leave_one_out_support_features():
    sigs = [["a", "b"], ["b", "c"], ["a", "c"]]
    df = STR.document_frequency(sigs)
    assert df == {"a": 2, "b": 2, "c": 2}
    loo = STR.leave_one_out_support_row(sigs[0], df)
    assert loo["n_unseen_atoms"] == 0
    assert loo["rare_atom_frac"] == pytest.approx(1.0)
    assert loo["median_log1p_df"] == pytest.approx(np.log(2.0))
    full = STR.support_row(sigs[0], df)
    assert full["rare_atom_frac"] == pytest.approx(1.0)  # df=2 is inside [1, 4]
    assert full["unseen_atom_frac"] == pytest.approx(0.0)
    common = STR.support_row(["a", "b"], {"a": 5, "b": 6})
    assert common["rare_atom_frac"] == pytest.approx(0.0)
    assert common["unseen_atom_frac"] == pytest.approx(0.0)


def test_leave_one_out_rejects_foreign_dictionary():
    with pytest.raises(ValueError):
        STR.leave_one_out_support_row(["z"], {"a": 1})


def test_train_only_dictionary_guard(fixture_molecules):
    train = list(range(0, 6))
    valid = [6, 7, 8]
    support_train = STR.build_support(
        group_key="fixture",
        train_rows=train,
        query_splits={"valid": valid},
        molecules=fixture_molecules,
        include_loo=False,
    )
    assert support_train.df == STR.document_frequency(
        [fixture_molecules[i].atom_signatures for i in train]
    )
    support_all = STR.build_support(
        group_key="fixture",
        train_rows=list(range(len(fixture_molecules))),
        query_splits={"valid": valid},
        molecules=fixture_molecules,
        include_loo=False,
    )
    assert support_all.df != support_train.df
    assert support_train.summary["train_only"] is True
    assert support_train.summary["valid_or_test_used_to_build_dictionary"] is False


def test_support_features_do_not_depend_on_test_labels(fixture_molecules):
    train = [0, 1, 2, 3]
    test = [4, 5, 6]
    labels = np.asarray([0.1, 0.2, 0.3, 0.4, 9.0, 9.0, 9.0])

    def build(ys: np.ndarray):
        # ``ys`` is deliberately unused inside build_support: the guard is that
        # changing the labels cannot change any feature.
        return STR.build_support(
            group_key="fixture",
            train_rows=train,
            query_splits={"test": test},
            molecules=fixture_molecules,
            include_loo=False,
        )

    first = build(labels)
    labels[test] += 1000.0
    second = build(labels)
    assert first.query.equals(second.query)
    assert first.summary["signature_dictionary_sha256"] == second.summary["signature_dictionary_sha256"]
    assert not any(name in first.query.columns for name in ("y", "target", "label"))


def test_support_query_alignment(fixture_molecules):
    support = STR.build_support(
        group_key="fixture",
        train_rows=[0, 1],
        query_splits={"valid": [2, 3, 4], "test": [5, 6]},
        molecules=fixture_molecules,
        include_loo=False,
    )
    assert np.array_equal(
        support.query[support.query["split"] == "valid"]["row_id"].to_numpy(),
        np.asarray([2, 3, 4]),
    )
    assert np.array_equal(
        support.query[support.query["split"] == "test"]["row_id"].to_numpy(),
        np.asarray([5, 6]),
    )


# ---------------------------------------------------------------------------
# scaffold folds and cluster bootstrap
# ---------------------------------------------------------------------------


def test_scaffold_folds_are_deterministic_and_group_whole_scaffolds():
    scaffolds = ["a", "b", "a", "c", "d", "b", "e", "f", "g", "h"]
    folds = STAT.deterministic_scaffold_folds(scaffolds, 5)
    order = np.asarray([7, 3, 0, 9, 8, 1, 5, 2, 6, 4])
    shuffled = STAT.deterministic_scaffold_folds(np.asarray(scaffolds)[order], 5)
    assert np.array_equal(folds[order], shuffled)
    for scaffold in set(scaffolds):
        assert len(set(folds[np.asarray(scaffolds) == scaffold])) == 1
    assert set(folds.tolist()) == set(range(5))


def test_cluster_index_groups_rows():
    clusters = STAT.ClusterIndex.from_values(["x", "x", "y", "z", "z"])
    assert clusters.n_clusters == 3
    assert clusters.profile()["max_cluster_size"] == 2


# ---------------------------------------------------------------------------
# frozen artefacts
# ---------------------------------------------------------------------------


def test_frozen_group_a_loads_and_passes_all_checks(frozen_group_a):
    assert frozen_group_a.key == "A_random42"
    assert set(frozen_group_a.rows) == {"train", "valid", "test"}
    assert len(frozen_group_a.rows["valid"]) == 420
    assert all(check["passed"] for check in frozen_group_a.audit["checks"])
    assert frozen_group_a.predictions["lcrc_a_r3_s0"].rows.tolist() == frozen_group_a.rows["valid"].tolist()
    assert frozen_group_a.predictions["pna_native_s0"].rows.tolist() == frozen_group_a.rows["valid"].tolist()
    # Test predictions were never loaded; only row ids and split metadata exist.
    assert "test" not in frozen_group_a.predictions


def test_frozen_group_b_loads_and_passes_all_checks(frozen_group_b):
    assert frozen_group_b.key == "B_lipo_random122"
    assert len(frozen_group_b.rows["valid"]) == 420
    assert len(frozen_group_b.rows["test"]) == 420
    assert all(check["passed"] for check in frozen_group_b.audit["checks"])
    for key, pred in frozen_group_b.predictions.items():
        split = "test" if key.endswith("_test") else "valid"
        assert pred.rows.tolist() == frozen_group_b.rows[split].tolist(), key
        assert np.all(np.isfinite(pred.pred)), key
        assert np.array_equal(pred.y, np.asarray(frozen_group_b.labels[split])), key


def test_random42_and_random122_are_distinct(frozen_group_a, frozen_group_b):
    sep = SRC.separation_record(frozen_group_a, frozen_group_b)
    assert sep["same_valid_row_set"] is False
    assert sep["same_train_row_set"] is False
    assert sep["treated_as_same_evaluation_set"] is False
    assert sep["per_molecule_pairing_across_groups"] is False


def test_analysis_outputs_are_self_consistent():
    analysis = C.ANALYSIS
    if not (analysis / "manifest.json").exists():
        pytest.skip("analysis outputs not generated yet")
    manifest = json.loads((analysis / "manifest.json").read_text())
    integrity = json.loads((analysis / "integrity_checks.json").read_text())
    assert manifest["flags"]["gpu_used"] is False
    assert manifest["flags"]["training_or_finetuning"] is False
    assert manifest["flags"]["group_a_and_group_b_pooled"] is False
    for name, record in manifest["outputs"].items():
        path = analysis / name
        assert path.exists(), name
        assert C.sha256_file(path) == record["sha256"], name
    assert integrity["all_blocking_checks_passed"] is True
    per_molecule = analysis / "per_molecule.csv"
    assert per_molecule.exists()
    import pandas as pd

    frame = pd.read_csv(per_molecule)
    assert len(frame) == 420 + 420 + 420
    assert set(frame["group"]) == {"A_random42", "B_lipo_random122"}
    assert set(frame[frame["group"] == "A_random42"]["split"]) == {"valid"}
    assert set(frame[frame["group"] == "B_lipo_random122"]["split"]) == {"valid", "test"}
    assert frame[frame["split"] == "test"]["exposure"].unique().tolist() == [
        "POST_HOC_EXPOSED_DISCOVERY"
    ]
    for column in (
        "unseen_atom_frac",
        "rare_atom_frac",
        "median_log1p_df",
        "p10_log1p_df",
        "max_train_tanimoto",
        "heavy_atoms",
        "tpsa",
    ):
        assert frame[column].notna().all(), column
    # group A must have the paired diagnosis columns; group B the candidate columns
    assert frame[frame["group"] == "A_random42"]["d_i"].notna().all()
    assert frame[frame["group"] == "B_lipo_random122"]["pred_combo_s0"].notna().all()
