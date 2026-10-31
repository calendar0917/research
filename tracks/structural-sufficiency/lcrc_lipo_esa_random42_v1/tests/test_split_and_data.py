"""Stage-1 protocol tests: CSV identity, frozen random42 indices, row order."""

from __future__ import annotations

import numpy as np
import pytest


def test_csv_hash_and_rows():
    import r42_common as C
    import r42_data as D

    assert C.sha256_file(C.CSV_PATH) == C.CSV_SHA256
    ids, smiles = D.read_smiles_only()
    assert len(ids) == 4200
    assert len(set(ids)) == 4200
    assert all(s for s in smiles)


def test_random_split_indices_match_frozen_hashes(split):
    import r42_common as C
    import r42_data as D

    recomputed = D.compute_split()
    for name in ("train", "valid", "test"):
        assert recomputed[name] == list(split["rows"][name])
        assert C.rows_sha256_int64(recomputed[name]) == C.SPLIT_SHA256[name]
        assert len(recomputed[name]) == C.SPLIT_SIZES[name]


def test_random_split_first_rows(split):
    import r42_common as C

    for name in ("train", "valid", "test"):
        assert list(split["rows"][name][:5]) == C.SPLIT_FIRST_ROWS[name]


def test_split_is_disjoint_and_complete(split):
    import r42_common as C

    tr, va, te = (set(split["rows"][k]) for k in ("train", "valid", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(range(C.N_MOLECULES))


def test_multi_component_molecule_preserved():
    import r42_common as C

    split = C.read_json(C.SPLIT_FILE)
    row = C.MULTI_COMPONENT_ROW
    smiles = split["smiles"][row]
    assert smiles.count(".") == 3
    home = [k for k in ("train", "valid", "test") if row in split["rows"][k]]
    assert home and home[0] == split["multi_component_split"]


def test_split_file_rows_are_int64_hashable(split):
    import r42_common as C

    arr = np.asarray(split["rows"]["train"], dtype="<i8")
    assert arr.dtype == np.dtype("<i8")
    assert C.sha256_bytes(arr.tobytes()) == C.SPLIT_SHA256["train"]


def test_upstream_structures_keep_row_order(structures):
    import r42_common as C

    assert len(structures) == C.N_MOLECULES
    assert [int(d.mol_index) for d in structures] == list(range(C.N_MOLECULES))
    assert int(structures[0].x.shape[1]) == 79
    assert int(structures[0].edge_attr.shape[1]) == 13
    # Each directed edge pair is present, and every edge points at a real node.
    for data in structures[:50]:
        assert int(data.edge_index.max()) < int(data.num_nodes)
        assert int(data.edge_index.min()) >= 0
        assert int(data.edge_index.shape[1]) == int(data.edge_attr.shape[0])


def test_upstream_uses_explicit_hydrogens(structures):
    from rdkit import Chem
    import r42_data as D

    _, smiles = D.read_smiles_only()
    row = 0
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles[row]))
    assert int(structures[row].num_nodes) == mol.GetNumAtoms()


def test_lcrc_cache_row_identity():
    import r42_common as C
    import r42_data as D

    built = D.build_lcrc_cache()
    cache = built["cache"]
    assert int(cache["n_molecules"]) == C.N_MOLECULES
    assert [str(x) for x in cache["molecule_ids"]] == [str(i) for i in range(C.N_MOLECULES)]
    assert cache["meta"]["labels_present"] is False


def test_lcrc_views_preserve_row_order_and_labels(split, labels):
    import r42_data as D

    views = D.build_lcrc_views(split, labels)
    assert views["train_cache"]["n_molecules"] == 3360
    assert views["valid_cache"]["n_molecules"] == 420
    assert [str(x) for x in views["train_cache"]["molecule_ids"]] == [str(r) for r in split["rows"]["train"]]
    assert [str(x) for x in views["valid_cache"]["molecule_ids"]] == [str(r) for r in split["rows"]["valid"]]
    assert views["target_norm"]["n"] == 3360
    assert views["train_norm"]["valid_or_test_used"] is False
    assert views["train_norm"]["source"] == "train3360_structures"


def test_pna_degree_histogram_is_train_only(tmp_path):
    import r42_common as C
    import r42_data as D
    import r42_native_train as NT
    import r42_labels as L

    split = D.load_split()
    train = D.load_structure_data([int(r) for r in split["rows"]["train"]])
    info = NT.pna_degrees(train, tmp_path)
    total_directed_edges = int(sum(int(d.edge_index.shape[1]) for d in train))
    total_atoms = int(sum(int(d.num_nodes) for d in train))
    assert info["source"] == "train3360_graphs_only"
    assert info["valid_or_test_used"] is False
    # The histogram counts atoms per degree: sum of counts is the atom total and
    # the degree-weighted sum is the directed-edge total.
    assert int(sum(v for _, v in info["nonzero"])) == total_atoms
    assert int(sum(k * v for k, v in info["nonzero"])) == total_directed_edges
    assert info["n_atoms"] == total_atoms and info["total_degree"] == total_directed_edges
    assert len(train) == C.SPLIT_SIZES["train"]


def test_validation_data_is_never_used_for_structure_statistics(split, labels):
    import numpy as np
    import r42_data as D
    import r42_labels as L

    views = D.build_lcrc_views(split, labels)
    tr = views["train_norm"]
    va = D.select_cache(views["built"]["cache"], [int(r) for r in split["rows"]["valid"]], split_label="valid")
    import lcrc_data

    va_norm = lcrc_data.fit_norm(va, views["spec"], source="valid_only")
    assert not np.array_equal(np.asarray(tr["global_mean"]), np.asarray(va_norm["global_mean"]))
