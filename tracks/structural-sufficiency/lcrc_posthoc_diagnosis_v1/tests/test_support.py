import numpy as np

import ph_structures as S


def test_support_counts_molecules_not_atoms():
    # Ethane: both carbons share the same rooted WL2 fingerprint.  One molecule
    # must contribute 1, not 2, to the support count.
    support, meta = S.train_support(["CC"])
    assert meta["n_train_molecules"] == 1
    assert max(support.values()) == 1


def test_two_molecules_increment_support():
    support, _ = S.train_support(["CC", "CC"])
    assert max(support.values()) == 2


def test_wl2_hash_is_stable_and_atom_indexed():
    mol = S.parse_smiles("CCO")[0]
    a = S.wl2_root_colors(mol)
    b = S.wl2_root_colors(mol)
    assert a == b
    assert len(a) == mol.GetNumAtoms()
    assert all(isinstance(x, str) and len(x) == 64 for x in a)


def test_unsanitized_fallback():
    smi = "O=C1O[Al]23(OC1=O)(OC(=O)C(=O)O2)OC(=O)C(=O)O3"
    mol, mode = S.parse_smiles(smi)
    assert mode == "unsanitized_fallback"
    colors = S.wl2_root_colors(mol)
    assert len(colors) == mol.GetNumAtoms()
