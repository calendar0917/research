from __future__ import annotations

import numpy as np

import cpu_gates as G
import lcrc_data as LD


def test_atom_permutation_invariance(spec, sample_graphs, sample_model):
    report = G.gate_permutation_invariance(sample_graphs, spec, sample_model, seed=7)
    assert report["status"] == "PASS"
    assert report["max_atom_feat_abs"] <= 1e-6
    assert report["max_scalar_abs"] <= 1e-6
    assert report["max_global_abs"] <= 1e-6
    assert report["max_y_abs"] <= 1e-5


def test_bond_endpoint_swap_and_order_invariance(spec, sample_graphs, sample_model):
    report = G.gate_bond_order_invariance(sample_graphs, spec, sample_model, seed=8)
    assert report["status"] == "PASS"
    assert report["max_y_abs"] <= 1e-6
    assert all(x["arrays_exact"] for x in report["per_graph"])


def test_path_record_order_invariance(spec, sample_graphs, sample_model):
    report = G.gate_path_order_invariance(sample_graphs, spec, sample_model, seed=9)
    assert report["status"] == "PASS"
    assert report["max_y_abs"] <= 1e-5


def test_single_vs_batch_and_index_isolation(spec, sample_cache, sample_model):
    rows = list(range(min(12, int(sample_cache["n_molecules"]))))
    report = G.gate_batch_invariance(sample_cache, spec, sample_model, rows=rows, seed=10)
    assert report["status"] == "PASS"
    assert report["max_single_vs_batch_abs"] <= 1e-6
    assert report["max_subset_vs_batch_abs"] <= 1e-6
    assert report["finite"] is True
    assert all(report["index_report"]["checks"].values())


def test_collate_does_not_mix_molecules(spec, sample_cache):
    rows = list(range(min(8, int(sample_cache["n_molecules"]))))
    batch = LD.collate(sample_cache, rows)
    atom_batch = batch["atom_batch"].numpy()
    bond_batch = batch["bond_batch"].numpy()
    bi = batch["bond_index"].numpy()
    assert np.array_equal(atom_batch[bi[:, 0]], bond_batch)
    assert np.array_equal(atom_batch[bi[:, 1]], bond_batch)
    assert np.array_equal(atom_batch[batch["path_end"].numpy()], batch["path_graph"].numpy())
    assert int(batch["n_graphs"]) == len(rows)
    assert int(batch["atom_off"][0]) == 0
