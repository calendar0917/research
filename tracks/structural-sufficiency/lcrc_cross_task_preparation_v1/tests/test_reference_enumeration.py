from __future__ import annotations

import networkx as nx
import numpy as np

import cpu_gates as G
import lcrc_data as LD
import molecular_graph as MG


def test_synthetic_edge_cases_match_independent_reference(spec, synthetic_graphs):
    report = G.gate_reference_enumeration(list(synthetic_graphs.values()), spec, label="synthetic")
    assert report["status"] == "PASS"
    assert report["all_equal"] is True


def test_real_samples_match_independent_reference(spec, sample_graphs):
    report = G.gate_reference_enumeration(sample_graphs, spec, label="real_sample")
    assert report["status"] == "PASS"
    assert report["all_equal"] is True


def test_edge_case_counts_and_components(spec, synthetic_graphs):
    expected = {
        "single_atom": (1, 0, 1, 0),
        "two_atom_single": (2, 1, 1, 0),
        "three_atom_path": (3, 2, 1, 2),
        "four_atom_linear": (4, 3, 1, 4),
        "triangle": (3, 3, 1, 6),
        "benzene": (6, 6, 1, 12),
        "branched": (4, 3, 1, 6),
        "charged": (2, 1, 1, 0),
        "two_components": (3, 1, 2, 0),
        "isolated_in_component": (3, 1, 2, 0),
        "unknown_atom_and_bond": (3, 2, 1, 2),
    }
    for name, (n_atoms, n_bonds, n_comp, n_paths) in expected.items():
        graph = synthetic_graphs[name]
        mol = LD.build_molecule(graph, spec)
        assert (graph.n_atoms, graph.n_bonds, graph.n_components) == (n_atoms, n_bonds, n_comp)
        assert len(mol["path_cat"]) == n_paths
        assert np.isfinite(np.asarray(mol["global"])).all()
        assert np.isfinite(np.asarray(mol["atom_scalar"])).all()


def test_unreachable_pairs_are_finite_and_not_counted_as_ge3(spec, synthetic_graphs):
    graph = synthetic_graphs["two_components"]
    mol = LD.build_molecule(graph, spec)
    names = [f.name for f in spec.global_feature_fields]
    idx = {n: i for i, n in enumerate(names)}
    g = np.asarray(mol["global"])
    # C-C plus an isolated N: 3 atoms, one finite pair, two unreachable pairs.
    assert g[idx["n_components"]] == 2
    assert g[idx["frac_unreachable_pairs"]] == 2 / 3
    assert g[idx["frac_finite_pairs_ge3"]] == 0
    assert g[idx["diameter_max_finite"]] == 1
    assert np.isfinite(g).all()


def test_ring_flags_agree_with_networkx_on_real_and_synthetic(spec, sample_graphs, synthetic_graphs):
    graphs = list(sample_graphs) + list(synthetic_graphs.values())
    for graph in graphs:
        analysis = MG.analyse_graph(graph)
        gx = nx.Graph()
        gx.add_nodes_from(range(graph.n_atoms))
        gx.add_edges_from(zip(graph.bond_u.tolist(), graph.bond_v.tolist()))
        cycle_edges = set()
        cycle_nodes = set()
        for cycle in nx.cycle_basis(gx):
            cycle_nodes.update(int(x) for x in cycle)
            for i in range(len(cycle)):
                cycle_edges.add(tuple(sorted((int(cycle[i]), int(cycle[(i + 1) % len(cycle)])))))
        assert set(int(x) for x in np.where(analysis.ring_atom)[0]) == cycle_nodes
        actual_edges = {
            tuple(sorted((int(u), int(v))))
            for u, v, flag in zip(graph.bond_u, graph.bond_v, analysis.is_ring_bond)
            if bool(flag)
        }
        assert actual_edges == cycle_edges
