"""Fast structural tests for the port-binding round (needs the local data caches)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import wp_common as C  # noqa: E402
import wp_data as D  # noqa: E402
import wp_model as M  # noqa: E402
import wp_probes as P  # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    return D.load_dataset("ogbg-mollipo")


def test_derived_pointers_are_consistent(dataset):
    words, derived = dataset["words"], dataset["derived"]
    for split in ("train", "valid"):
        pre = f"derived/{split}/"
        site_group = derived[pre + "site_group"]
        site_ptr = derived[pre + "site_ptr"]
        group_ptr = derived[pre + "word_group_ptr"]
        n_words = int(words[f"word/{split}/word_n_atoms"].shape[0])
        site_word = np.repeat(np.arange(n_words), np.diff(site_ptr))
        assert site_word.shape[0] == site_group.shape[0]
        assert group_ptr[-1] == derived[pre + "group_count"].shape[0]
        assert np.all(site_group >= group_ptr[site_word])
        assert np.all(site_group < group_ptr[site_word + 1])
        # every (p, a) site pair exists exactly once per word
        counts = derived[pre + "pp_ptr"]
        n = np.asarray(words[f"word/{split}/word_n_atoms"], dtype=np.int64)
        assert np.array_equal(np.diff(counts), n * n)


def test_pack_batch_indices_stay_local(dataset):
    split = D.make_split(dataset, "train")
    batch = D.pack_batch(split, np.arange(16))
    n_atoms = int(batch["atom_feat"].shape[0])
    n_bonds = int(batch["bond_feat"].shape[0])
    n_sites = int(batch["site_atom"].shape[0])
    n_words = int(batch["word_tmpl"].shape[0])
    assert batch["site_atom"].max() < n_atoms
    assert batch["edge_bond"].max() < n_bonds
    assert batch["edge_src"].max() < n_sites and batch["edge_dst"].max() < n_sites
    assert batch["inc_own_site"].max() < n_sites and batch["inc_other_site"].max() < n_sites
    assert batch["inc_own_atom"].max() < n_atoms and batch["inc_other_atom"].max() < n_atoms
    assert batch["pp_p"].max() < n_sites and batch["pp_a"].max() < n_sites
    assert batch["site_group"].max() < batch["n_groups"]
    assert batch["pair_a"].max() < batch["inc_own_site"].shape[0]
    assert batch["graph_of_word"].max() < batch["n_graphs"]
    assert batch["word_of_inc"].max() < n_words


def test_subset_pack_matches_full_pack(dataset):
    import torch

    split = D.make_split(dataset, "valid")
    words = dataset["words"]
    tmpl = {k[len("tmpl/"):]: v for k, v in words.items() if k.startswith("tmpl/")}
    tmpl["dist"] = dataset["derived"]["tmpl/dist"]
    model = M.PortBindModel(M.ARMS["BIND_REL"], tmpl, C.official_dims())
    M.init_by_name(model, 1000)
    model.eval()
    with torch.no_grad():
        full = model(D.to_device(D.pack_batch(split, np.arange(12)), "cpu"))["pred"].numpy()
        part = np.concatenate([
            model(D.to_device(D.pack_batch(split, np.arange(a, min(a + 4, 12))), "cpu"))["pred"].numpy()
            for a in range(0, 12, 4)])
    assert np.max(np.abs(full - part)) <= 1e-5


def test_all_arms_share_parameter_shape_and_init(dataset):
    import torch

    words = dataset["words"]
    tmpl = {k[len("tmpl/"):]: v for k, v in words.items() if k.startswith("tmpl/")}
    tmpl["dist"] = dataset["derived"]["tmpl/dist"]
    dims = C.official_dims()
    reference = None
    for spec in M.ARMS.values():
        model = M.PortBindModel(spec, tmpl, dims)
        M.init_by_name(model, 1000)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        if reference is None:
            reference = (shapes, state)
        else:
            assert shapes == reference[0]
            for key, value in state.items():
                assert torch.equal(value, reference[1][key]), key
        assert M.count_parameters(model) <= C.load_config()["model"]["param_cap"]


def test_probe_a1_pair_is_symmetric_under_mean():
    report = P.a1_probe()
    assert report["p_q_attribute_rows_equal"]
    assert report["mean_equal_input_pairs_gap_le_1e5"]
    assert report["bind_better_than_mean"]


def test_probe_a2_given_pair():
    report = P.a2_probe()
    assert report["given_pair_same_first_order_signature"]
    assert all(v["matches"] for v in report["given_pairs"].values())
    assert report["self_invariant_ok"]
    assert report["rel_fits_better"]
