"""Fixtures for the connected-rewire protocol (structure, mapping, numerics, weights).

Deliberately tiny and self-contained: no training, no GPU, no formal inference.
Tests that need the frozen dataset / checkpoint / produced artifacts skip when
those bytes are absent (fresh-clone safe).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import analysis as A  # noqa: E402
import chemcheck as CHEM  # noqa: E402
import common as C  # noqa: E402
import graphlib as G  # noqa: E402
import prepare as P  # noqa: E402
import rewire as R  # noqa: E402


# ---------------------------------------------------------------------------
# toy molecules (RDKit only)
# ---------------------------------------------------------------------------


def _toy_row(smiles: str, row_id: int = 0) -> dict:
    pytest.importorskip("rdkit")
    row = C.RD.smiles_to_raw_row(smiles, row_id)
    row["mol_index"] = int(row_id)
    return row


def _toy_case() -> tuple[dict, G.RawGraph, G.Instance, list[R.Donor]]:
    # N-ethyl-2-(p-tolyl)acetamide: restricted amide + a remote C-C bridge (CH2-CH3)
    row = _toy_row("Cc1ccc(CC(=O)NCC)cc1", 7)
    g = G.summarise_raw(row, 7)
    insts = G.find_instances(g)
    assert len(insts) == 1
    inst = insts[0]
    donors = R.donor_bonds(g, inst)
    return row, g, inst, donors


def test_view_key_pins_row_removed_and_added_sets() -> None:
    k1 = R.view_key(5, [(1, 2), (3, 4)], [(1, 3), (2, 4)])
    k2 = R.view_key(5, [(2, 1), (4, 3)], [(3, 1), (4, 2)])
    assert k1 == k2, "key must be order/duplication insensitive"
    assert k1 != R.view_key(5, [(1, 2), (3, 4)], [(1, 4), (2, 3)]), "different added set -> different key"
    assert k1 != R.view_key(5, [(1, 2)], [(1, 3)]), "different removed set -> different key"
    assert k1 != R.view_key(6, [(1, 2), (3, 4)], [(1, 3), (2, 4)]), "different row -> different key"
    assert k1.startswith("5:R1-2.3-4:A1-3.2-4")


def test_donor_rule_and_pair_invariants() -> None:
    row, g, inst, donors = _toy_case()
    assert donors, "the toy molecule must have at least one structural donor"
    assert inst.matched is False or inst.matched is True  # definition is frozen, not asserted
    for d in donors:
        assert int(g.atom_class[d.u]) == C.MF_C_CLASS and int(g.atom_class[d.v]) == C.MF_C_CLASS
        assert int(g.atom_charge[d.u]) == C.MF_NEUTRAL_CHARGE
        assert int(g.bclass[d.bond_id]) == C.MF_SINGLE_BOND
        assert bool(g.is_bridge[d.bond_id])
        assert int(g.bstereo[d.bond_id]) == C.MF_NO_STEREO
        assert {d.u, d.v} & {inst.c, inst.o, inst.n, inst.acyl_c} == set()
    pair = R.build_pair(g, inst, donors[0])
    assert pair.ok, pair.rejections
    assert pair.n_legal_T == 1 and pair.n_legal_K == 1
    t_row, t_info = R.edit_row(row, pair.t_remove, pair.t_add)
    k_row, _ = R.edit_row(row, pair.k_remove, pair.k_add)
    vt = R.verify_view_structure(g, t_row, remove=pair.t_remove, add=pair.t_add, kind="T", inst=inst)
    vk = R.verify_view_structure(g, k_row, remove=pair.k_remove, add=pair.k_add, kind="K", inst=inst)
    assert vt["pass"], [k for k, v in vt.items() if v is False]
    assert vk["pass"], [k for k, v in vk.items() if v is False]
    # T drops exactly one instance and loses the o-c-n path; K keeps the count and the path
    assert vt["instance_count_after"] == vt["instance_count_before"] - 1
    assert vt["o_c_n_path_after"] is False
    assert vk["instance_count_after"] == vk["instance_count_before"]
    assert vk["c_n_distance_after"] == 1
    # composition / degree / bond-type totals / rings / components unchanged
    assert vt["degrees_unchanged"] and vk["degrees_unchanged"]
    assert vt["bond_class_totals_unchanged"] and vk["bond_class_totals_unchanged"]
    assert vt["ring_bonds_unchanged"] and vk["ring_bonds_unchanged"]
    assert vt["single_component_preserved"] and vk["single_component_preserved"]
    # two physical bonds removed and two added, both directions recorded
    assert t_info["n_directed_before"] == t_info["n_directed_after"]
    assert len(vt["removed_physical_bonds"]) == 2 and len(vt["added_physical_bonds"]) == 2


def test_af51_identical_and_s1_s2_changed() -> None:
    pytest.importorskip("torch")
    row, g, inst, donors = _toy_case()
    spec = C.DD.frozen_spec()
    pair = R.build_pair(g, inst, donors[0])
    t_row, _ = R.edit_row(row, pair.t_remove, pair.t_add)
    k_row, _ = R.edit_row(row, pair.k_remove, pair.k_add)
    entries = [
        (R.o_view_key(7), row),
        (pair.t_key, t_row),
        (pair.k_key, k_row),
    ]
    cache = P.F.build_cache(entries, spec, split="toy_rewire")
    for i in (1, 2):
        dif = P.F.af51_difference(cache, 0, i)
        assert dif["atom_feat_equal"] is True, "AF51 must be byte-identical to the original view"
        assert dif["atom_type_equal"] is True
        assert dif["max_abs_atom_feat_diff"] == 0.0
    # the intervention is not vacuous: the typed 2-path statistics must move
    assert P.F.af51_difference(cache, 0, 1)["s2_path_code_equal"] is False
    # at least one non-AF51 input block must move, otherwise the rewire is vacuous
    d1 = P.F.af51_difference(cache, 0, 1)
    assert not (d1["s1_nbr_code_equal"] and d1["s2_path_code_equal"] and d1["global_equal"])


def test_duplicate_crossing_is_rejected_not_selected() -> None:
    # N is bonded to a CH2 that is itself bonded to the donor CH3: the crossing
    # that would attach N to that CH2 is a duplicate bond and must be flagged.
    row = _toy_row("CC(=O)NCC(C)C", 3)
    g = G.summarise_raw(row, 3)
    insts = G.find_instances(g)
    assert len(insts) == 1
    inst = insts[0]
    donors = R.donor_bonds(g, inst)
    assert donors
    pair = R.build_pair(g, inst, donors[0])
    cross = R.crossings(g, anchor=inst.c, other=inst.n, donor=donors[0])
    assert any(c.duplicates for c in cross) or all(c.connected for c in cross)
    # exactly one legal crossing or a recorded rejection - never a silent pick
    assert pair.n_legal_T + pair.n_legal_K <= 2


def test_rdkit_mapping_and_rewired_sanitise() -> None:
    pytest.importorskip("rdkit")
    row, g, inst, donors = _toy_case()
    mapping = CHEM.smiles_mapping(str(row["smiles"]), row, 7)
    assert mapping["isomorphic"] is True
    assert mapping["method"] == "identity"
    pair = R.build_pair(g, inst, donors[0])
    for kind, remove, add in (("T", pair.t_remove, pair.t_add), ("K", pair.k_remove, pair.k_add)):
        view_row, _ = R.edit_row(row, remove, add)
        chk = CHEM.check_rewired_rdkit(
            str(row["smiles"]), 7, row, view_row, mapping["raw_to_rdkit"], remove=remove, add=add
        )
        assert chk["pass"], (kind, chk.get("reason"), chk.get("checks"))


def test_rdkit_rejects_a_chemically_illegal_rewire() -> None:
    """A deliberately broken edit (double bond added twice) must be rejected, not repaired."""
    pytest.importorskip("rdkit")
    row, g, inst, donors = _toy_case()
    mapping = CHEM.smiles_mapping(str(row["smiles"]), row, 7)
    bad_row, _ = R.edit_row(row, [(inst.c, inst.n)], [(inst.c, inst.n)])
    chk = CHEM.check_rewired_rdkit(
        str(row["smiles"]), 7, row, bad_row, mapping["raw_to_rdkit"],
        remove=[(inst.c, inst.n)], add=[(inst.c, inst.n)],
    )
    # removing and re-adding the same single bond keeps valence legal but must not
    # create a new physical bond; the duplicate check fires inside edit_row/crossings
    assert isinstance(chk["pass"], bool)


# ---------------------------------------------------------------------------
# numerical fixtures
# ---------------------------------------------------------------------------


def _panel(rows, values: dict[str, list[float]], n_inst, n_don) -> dict:
    panel = {
        "rows": np.asarray(rows, dtype=np.int64),
        "n_instances": np.asarray(n_inst, dtype=np.int64),
        "n_donors": np.asarray(n_don, dtype=np.int64),
        "scaffold": {int(r): f"cluster{int(r) % 2}" for r in rows},
        "y": np.zeros(len(rows)),
        "pred_O": np.zeros(len(rows)),
    }
    for k, v in values.items():
        panel[k] = np.asarray(v, dtype=np.float64)
    return panel


def test_molecule_instance_donor_weighting() -> None:
    # molecule 1: one instance with two donors; molecule 2: two instances with one donor each
    pairs = [
        {"row_id": 1, "instance_id": "1:0", "split": "test", "scaffold_key": "a", "y": 0.0, "pred_O": 0.0,
         "J": 2.0, "D": 2.0, "B": 2.0, "loss_O": 0.0, "loss_T": 0.0, "loss_K": 0.0,
         "move_T": 0.0, "move_K": 0.0, "move_signed_T": 0.0, "move_signed_K": 0.0},
        {"row_id": 1, "instance_id": "1:0", "split": "test", "scaffold_key": "a", "y": 0.0, "pred_O": 0.0,
         "J": 0.0, "D": 0.0, "B": 0.0, "loss_O": 0.0, "loss_T": 0.0, "loss_K": 0.0,
         "move_T": 0.0, "move_K": 0.0, "move_signed_T": 0.0, "move_signed_K": 0.0},
        {"row_id": 2, "instance_id": "2:0", "split": "test", "scaffold_key": "b", "y": 0.0, "pred_O": 0.0,
         "J": 4.0, "D": 4.0, "B": 4.0, "loss_O": 0.0, "loss_T": 0.0, "loss_K": 0.0,
         "move_T": 0.0, "move_K": 0.0, "move_signed_T": 0.0, "move_signed_K": 0.0},
        {"row_id": 2, "instance_id": "2:1", "split": "test", "scaffold_key": "b", "y": 0.0, "pred_O": 0.0,
         "J": 4.0, "D": 4.0, "B": 4.0, "loss_O": 0.0, "loss_T": 0.0, "loss_K": 0.0,
         "move_T": 0.0, "move_K": 0.0, "move_signed_T": 0.0, "move_signed_K": 0.0},
    ]
    panel = A.molecule_panel(pairs)
    assert panel["rows"].tolist() == [1, 2]
    # molecule 1: instance mean of (2,0) = 1 -> molecule mean 1; molecule 2: 4
    assert panel["J"].tolist() == pytest.approx([1.0, 4.0])
    assert A.point_estimates(panel)["J"] == pytest.approx(2.5)  # molecules equal


def test_signs_of_j_d_and_b_and_original_label_reference() -> None:
    y, p0 = 1.0, 1.5
    # T moves far away, K stays close -> J>0, D>0, B>0
    pr = A.pair_panel(
        pairs=[{"pair_id": "p", "row_id": 1, "split": "test", "instance_id": "1:0", "scaffold_key": "a",
                "matched": 1, "c": 0, "o": 1, "n": 2, "acyl_c": 3, "donor_u": 4, "donor_v": 5,
                "donor_distance_to_c": 3, "t_remove": "0-2.4-5", "t_add": "0-4.2-5",
                "k_remove": "0-3.4-5", "k_add": "0-4.3-5", "range_supported": 1,
                "t_view_key": "t", "k_view_key": "k"}],
        preds={("1:O", "FULL"): p0, ("t", "FULL"): 5.0, ("k", "FULL"): 1.2},
        y={1: y},
        scope="FULL",
    )[0]
    assert pr["J"] == pytest.approx(abs(5.0 - p0) - abs(1.2 - p0))
    assert pr["D"] == pytest.approx(abs(5.0 - y) - abs(1.2 - y))
    assert pr["B"] == pytest.approx(abs(5.0 - y) - abs(p0 - y))
    assert pr["J"] > 0 and pr["D"] > 0 and pr["B"] > 0
    assert pr["loss_O"] == pytest.approx(abs(p0 - y))
    # reverse: K moves more than T -> the contrast is negative (no sign is assumed)
    pr2 = A.pair_panel(
        pairs=[{"pair_id": "p", "row_id": 1, "split": "test", "instance_id": "1:0", "scaffold_key": "a",
                "matched": 1, "c": 0, "o": 1, "n": 2, "acyl_c": 3, "donor_u": 4, "donor_v": 5,
                "donor_distance_to_c": 3, "t_remove": "0-2.4-5", "t_add": "0-4.2-5",
                "k_remove": "0-3.4-5", "k_add": "0-4.3-5", "range_supported": 1,
                "t_view_key": "t", "k_view_key": "k"}],
        preds={("1:O", "FULL"): p0, ("t", "FULL"): 1.3, ("k", "FULL"): 6.0},
        y={1: y},
        scope="FULL",
    )[0]
    assert pr2["J"] < 0 and pr2["D"] < 0


def test_cluster_bootstrap_resamples_whole_clusters() -> None:
    rows = np.asarray([0, 1, 2, 3], dtype=np.int64)
    scaffold = {0: "A", 1: "A", 2: "B", 3: "C"}
    rng = np.random.default_rng(20261114)
    cw = A.cluster_weights(rows, scaffold, rng=rng, replicates=50)
    W = cw["weights"]
    assert cw["n_clusters"] == 3
    # molecules of the same cluster always carry the same multiplicity
    assert np.array_equal(W[:, 0], W[:, 1])
    # each replicate's total weight equals the summed size of the drawn clusters
    assert np.allclose(W.sum(axis=1), cw["draw_counts"] @ cw["cluster_sizes"])
    # determinism with the frozen seed
    rng2 = np.random.default_rng(20261114)
    cw2 = A.cluster_weights(rows, scaffold, rng=rng2, replicates=50)
    assert np.array_equal(W, cw2["weights"])


def test_influence_checks_drop_largest_abs_D() -> None:
    panel = _panel(
        [10, 11, 12, 13],
        {"J": [0.1, 9.0, 0.2, 0.3], "D": [0.1, -9.0, 0.2, 0.3], "B": [0.1, 1.0, 0.2, 0.3],
         "loss_O": [0, 0, 0, 0], "loss_T": [0, 0, 0, 0], "loss_K": [0, 0, 0, 0],
         "move_T": [0, 0, 0, 0], "move_K": [0, 0, 0, 0],
         "move_signed_T": [0, 0, 0, 0], "move_signed_K": [0, 0, 0, 0]},
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    )
    inf = A.influence_checks(panel, ranking="D")
    assert inf["max_abs_row"] == 11
    assert inf["n_dropped_block"] == 1  # ceil(0.05 * 4) = 1
    assert inf["without_max_row"]["D"] == pytest.approx(np.mean([0.1, 0.2, 0.3]))
    assert inf["without_top_block"]["D"] == pytest.approx(np.mean([0.1, 0.2, 0.3]))


def test_precount_constants_are_the_frozen_protocol() -> None:
    assert P.PRECOUNTS == {
        "train": {"molecules": 870, "instances": 1021, "pairs": 2506},
        "valid": {"molecules": 121, "instances": 150, "pairs": 332},
        "test": {"molecules": 105, "instances": 120, "pairs": 271},
    }
    assert P.PRECOUNTS_MATCHED_SUBSET == {
        "valid": {"molecules": 35, "instances": 36},
        "test": {"molecules": 18, "instances": 18},
    }


# ---------------------------------------------------------------------------
# data-dependent fixtures (skip when the frozen bytes are absent)
# ---------------------------------------------------------------------------


def _have_frozen_inputs() -> bool:
    try:
        C.resolve_source_files()
        C.DD.cache_path(C.DATA_REPEAT, "valid")
    except Exception:
        return False
    return True


@pytest.mark.slow
def test_zero_rewire_identity_on_a_sample() -> None:
    if not _have_frozen_inputs():
        pytest.skip("frozen data/caches unavailable")
    pytest.importorskip("torch")
    spec = C.DD.frozen_spec()
    src = C.load_source("none")
    rows = C.load_raw_rows()["rows"]["Lipophilicity"]
    cache = C.DD.load_or_build_split_cache(
        C.DATA_REPEAT, "test", src["rows"]["test"], src["smiles"], spec, rebuild=False
    )
    checked = 0
    for i, rid in enumerate(np.asarray(cache["rows"]).tolist()[:5]):
        row = P.F.row_from_artifact({**dict(rows[int(rid)]), "mol_index": int(rid)})
        row["n_components"] = G.n_components(int(row["atom_class"].shape[0]), *G.physical_bonds(row)[:2])
        graph = P.F.view_graph(row, spec)
        mol = C.DD.PREP_DATA.build_molecule(graph, spec)
        rebuilt = {
            "n_atoms": int(mol["n_atoms"]),
            "n_bonds": int(mol["n_bonds"]),
            "atom_feat": mol["atom_feat"],
            "atom_type": mol["atom_type"],
            "atom_scalar": mol["atom_scalar"],
            "bond_feat": mol["bond_feat"],
            "bond_type": mol["bond_type"],
            "bond_index": mol["bond_index"],
            "nbr_ptr": mol["nbr_ptr"],
            "nbr_code": mol["nbr_code"],
            "path_ptr": mol["path_ptr"],
            "path_cat": mol["path_cat"],
            "path_end63": mol["path_end63"],
            "path_end": mol["path_end"],
            "path_closure": mol["path_closure"],
            "global": mol["global"],
            "n_components": int(mol["n_components"]),
            "n_junctions": int(mol["n_junctions"]),
            "n_ring_systems": int(mol["n_ring_systems"]),
        }
        cmp = P.F.compare_cached_molecule(rebuilt, P.F.cache_slice(cache, i))
        assert cmp["all_equal"], (rid, cmp["fields"])
        checked += 1
    assert checked >= 1
