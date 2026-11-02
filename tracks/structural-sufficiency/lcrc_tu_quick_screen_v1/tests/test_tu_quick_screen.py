"""Data-free gates for the TU quick screen (synthetic graphs only)."""

from __future__ import annotations

import numpy as np

import tu_adapter as A
import tu_common as C
import tu_gates as G


def _assert_subchecks_ok(report: dict, keys: list[str]) -> None:
    for key in keys:
        assert report[key]["ok"], (key, report[key])


def test_hand_computed_structures():
    report = G.gate_hand_computed()
    _assert_subchecks_ok(
        report,
        ["chain", "triangle", "isolated_node", "disconnected", "unseen_label_maps_to_unk", "degree_buckets"],
    )


def test_permutation_invariance():
    report = G.gate_permutation()
    assert report["ok"], report
    assert report["details"]["max_abs_logit_delta"] < 1e-5


def test_batching_and_shapes():
    report = G.gate_batching_and_shapes()
    assert report["ok"], report


def test_shared_init_and_deleted_block():
    report = G.gate_shared_init()
    assert report["ok"], report
    assert report["details"]["deleted_columns"] == 96
    assert report["details"]["param_diff"] == 18432


def test_gradients_reach_every_used_branch():
    report = G.gate_gradients()
    assert report["ok"], report
    for name in ("FULL", "C1", "GIN"):
        assert report["checks"][name]["ok"], report["checks"][name]


def test_edge_cleaning_counts():
    # 0-1 and 1-2 listed twice, 0-2 listed once, self-loop 2-2 once
    ei = np.asarray([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 2]], dtype=np.int64)
    out = A.clean_edge_index(ei, 3)
    assert out["pairs"].tolist() == [[0, 1], [0, 2], [1, 2]]
    assert out["n_directed_records"] == 6
    assert out["n_self_loops_dropped"] == 1
    assert out["n_physical_edges"] == 3
    assert out["n_duplicate_directed_records_merged"] == 2
    assert out["n_pairs_with_odd_multiplicity"] == 1


def test_spec_widths_and_vocabularies():
    label_types = A.node_type_table("label", [0, 1])
    spec = A.build_tu_spec("NCI1", label_types)
    k = spec.n_atom_types
    assert k == 3
    assert spec.atom_unk_index == 2
    assert spec.atom_feat_dim == k + 29
    assert spec.bond_feat_dim == 2
    assert spec.n_scalars == 9
    assert spec.global_dim == 28
    assert spec.n_bond_types == 2
    assert spec.n_nbr_codes == 2 * k
    assert spec.n_path_codes == 4 * k
    assert spec.bond_unk_id == 2
    degree_types = A.node_type_table("degree", None)
    dspec = A.build_tu_spec("IMDB-MULTI", degree_types)
    assert dspec.n_atom_types == 18
    assert dspec.atom_type_names[-1] == "UNK"
    assert dspec.atom_type_names[-2] == "DEG_16PLUS"
    assert dspec.atom_feat_dim == 18 + 29


def test_split_partition_and_determinism():
    rng = np.random.default_rng(0)
    y = np.asarray([0] * 600 + [1] * 400, dtype=np.int64)
    split = A.build_split(y, 800, 100, 100)
    union = np.concatenate([split["train"], split["valid"], split["test"]])
    assert np.unique(union).size == y.size
    assert split["train"].size == 800 and split["valid"].size == 100 and split["test"].size == 100
    assert np.intersect1d(split["train"], split["test"]).size == 0
    again = A.build_split(y, 800, 100, 100)
    for name in ("train", "valid", "test"):
        assert np.array_equal(split[name], again[name])
        assert C.index_hash(split[name]) == C.index_hash(again[name])
    # stratification keeps both classes in every split
    for name in ("train", "valid", "test"):
        assert set(np.unique(y[split[name]]).tolist()) == {0, 1}


def test_local_layout_offsets():
    import tu_model as M

    spec = A.build_tu_spec("PROTEINS", A.node_type_table("label", [0, 1, 2]))
    full = M.local_layout(spec, "FULL")
    c1 = M.local_layout(spec, "C1")
    assert full["a"] == (0, 96)
    assert full["b_sum"] == (96, 192)
    assert full["b_mean"] == (192, 288)
    assert c1["s1_joint"][0] == 192
    assert full["atom_scalar"][1] == 529
    assert c1["atom_scalar"][1] == 433
    assert len(M.local_columns(spec, "FULL")) == 529
    assert len(M.local_columns(spec, "C1")) == 433
