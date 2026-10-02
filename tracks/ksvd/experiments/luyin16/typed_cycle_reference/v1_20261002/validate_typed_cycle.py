"""Information and implementation audit on synthetic typed graphs, not ZINC."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import numpy as np

from typed_cycle_reference import (Graph, witness_pair, chordless_cycles, semantic110,
                                  dihedral_views, ring_latents, ring_mlp_init,
                                  ring_summary, relabel, distances, SHELL_PAIRS,
                                  square_double_statistics, RING_INPUT_DIM)
from latent_dictionary_bridge_reference import bridge, frame_initialization


def baseline_fused(graph, seed):
    """Source-derived Sem108 binding/slot interface, uniform structural orbit.

    Cube is vertex-transitive, so any invariant pure-topology phi65 gives one
    structural coordinate on all vertices. Its actual learned value may be
    arbitrary; use a random nonzero 33-vector to exercise the binding formula.
    Labels are deliberately absent from that coordinate, as in the repo.
    """
    rng = np.random.default_rng(seed)
    c = rng.normal(size=33)
    WA = rng.normal(scale=0.1, size=(33, 96))
    WAC = rng.normal(scale=0.1, size=(28, 96))
    WE = rng.normal(scale=0.1, size=(99, 48))
    WEC = rng.normal(scale=0.1, size=(4, 48))
    node_response = (c @ WA) * WAC[0] / np.sqrt(96)
    g = np.concatenate([2 * c, np.zeros(33), c * c])
    edge_responses = (g @ WE)[None, :] * WEC / np.sqrt(48)
    wn1, wn2 = rng.normal(scale=.1, size=(96, 64)), rng.normal(scale=.1, size=(64, 48))
    we1, we2 = rng.normal(scale=.1, size=(48, 48)), rng.normal(scale=.1, size=(48, 32))
    bn1, bn2 = rng.normal(scale=.1, size=64), rng.normal(scale=.1, size=48)
    be1, be2 = rng.normal(scale=.1, size=48), rng.normal(scale=.1, size=32)
    def mlp(x, w1, b1, w2, b2):
        z = x @ w1 + b1
        return (z / (1 + np.exp(-z))) @ w2 + b2
    dist = distances(graph)
    fused = []
    sem = semantic110(graph)
    for root, row in enumerate(dist):
        node = np.array([np.sum(row == shell) * node_response for shell in range(3)])
        edge = np.zeros((6, 48))
        for u, v, b in graph.edges:
            if row[u] <= 2 and row[v] <= 2:
                pair = tuple(sorted((int(row[u]), int(row[v]))))
                edge[SHELL_PAIRS.index(pair)] += edge_responses[b]
        fused.append(np.concatenate([sem[root], mlp(node, wn1, bn1, wn2, bn2).ravel(),
                                     mlp(edge, we1, be1, we2, be2).ravel()]))
    return np.stack(fused)


def closed_ring(atoms):
    n = len(atoms)
    edges = sorted((min(i, (i+1) % n), max(i, (i+1) % n), 0) for i in range(n))
    return Graph(tuple(atoms), tuple(edges))


def run():
    a, b = witness_pair()
    assert tuple((u, v) for u, v, _ in a.edges) == tuple((u, v) for u, v, _ in b.edges)
    assert np.array_equal(semantic110(a), semantic110(b))
    assert np.max(np.ptp(semantic110(a), axis=0)) == 0
    assert np.array_equal(np.bincount([x[2] for x in a.edges]),
                          np.bincount([x[2] for x in b.edges]))
    cycles_a, cycles_b = chordless_cycles(a), chordless_cycles(b)
    assert cycles_a == cycles_b and [len(c) for c in cycles_a] == [4] * 6 + [6] * 4
    report = dict(
        scope="synthetic typed-graph information audit; not molecules from ZINC, no PyTorch/MAE run",
        repository_revision="7fa46ab300b43c6457499e45408296c5a4bd4c5c",
        official_test_loaded=False,
        old_information=dict(same_untyped_graph=True, same_global_chemical_marginals=True,
                             semantic110_max_delta=0.0, same_cycle_spectrum=True,
                             uniform_structural_orbit="source-derived invariance argument; not a real phi65 execution"),
        square_double_counts=dict(A=square_double_statistics(a), B=square_double_statistics(b)),
        limitation="Different typed objects do not imply different penalized-logP labels. No target-relevance claim.",
    )
    report["square_double_squared_sum"] = dict(
        A=sum(x*x for x in square_double_statistics(a)),
        B=sum(x*x for x in square_double_statistics(b)))

    # Full previous NumPy task path includes both unary and pair moments.
    # This does not execute the repository's PyTorch feature builder.
    from existing_spine_reference import WidthSpec, state_init, forward, mlp
    pairs = np.array([(i, j) for i in range(8) for j in range(i+1, 8)], dtype=int).T
    buckets = np.minimum(distances(a)[pairs[0], pairs[1]]-1, 4)
    rng = np.random.default_rng(24)
    relations = rng.normal(size=(28, 15))  # arbitrary SAME chemistry-free topology relation
    relations[:, -1] = 0
    auxiliary = rng.normal(size=(1, 40))  # SAME global/topology output
    deltas = {}
    for m in [1, 3]:
        values = []
        for seed in [5, 13, 29]:
            fused_a, fused_b = baseline_fused(a, seed), baseline_fused(b, seed)
            assert np.max(abs(fused_a-fused_b)) < 1e-12
            state = state_init(WidthSpec(m), seed)
            kwargs = (np.zeros(8, dtype=int), pairs, buckets, relations, auxiliary)
            ya = forward(state, fused_a, *kwargs)
            yb = forward(state, fused_b, *kwargs)
            values.append(float(np.max(abs(ya-yb))))
        assert max(values) < 1e-12
        deltas[str(m)] = values
    report["old_source_derived_reference_prediction_deltas"] = deltas

    # Extra ring reader columns start at zero: preserve the existing function.
    state=state_init(WidthSpec(1),5)
    base_pred,base_aux=forward(state,baseline_fused(a,5),*kwargs,return_aux=True)
    reader_state=copy.deepcopy(state["reader"])
    reader_state["w1"]=np.concatenate([reader_state["w1"],np.zeros((97,13))],axis=0)
    ring_block=rng.normal(size=(1,97))
    enlarged=np.concatenate([base_aux["graph"],ring_block],axis=1)
    identity=(mlp(enlarged,reader_state) @ state["final_w"]+state["final_b"]).ravel()
    assert np.array_equal(identity,base_pred)
    report["zero_ring_reader_identity_max_delta"]=0.0

    state = ring_mlp_init(3)
    D, V = frame_initialization()
    Sa, Sb = ring_summary(a, state, D, V, bridge), ring_summary(b, state, D, V, bridge)
    linear = float(np.linalg.norm(ring_latents(a, state, linear=True).sum(0)
                                  - ring_latents(b, state, linear=True).sum(0)))
    nonlinear = float(np.linalg.norm(Sa-Sb))
    assert linear < 1e-12 and nonlinear > 1e-4
    report["iteration"] = dict(
        untyped_ring_counts="FAIL information gain: exact equality",
        linear_orbit_encoder_first_moment_delta=linear,
        typed_nonlinear_ring_dictionary_summary_delta=nonlinear)

    # Same ring composition, different cyclic atom arrangement.
    c = closed_ring([0, 1, 0, 1, 0, 1])
    d = closed_ring([0, 0, 0, 1, 1, 1])
    orbit_c, orbit_d = dihedral_views(c, chordless_cycles(c)[0]), dihedral_views(d, chordless_cycles(d)[0])
    assert np.array_equal(orbit_c.mean(0), orbit_d.mean(0))
    arrangement_delta = float(np.linalg.norm(ring_summary(c, state, D, V, bridge)
                                            - ring_summary(d, state, D, V, bridge)))
    assert arrangement_delta > 1e-4
    report["arrangement"] = dict(raw_orbit_mean_delta=0.0, nonlinear_dictionary_delta=arrangement_delta)

    # Exact orbit set, random relabeling, no cycles, disjoint union.
    invariance = []
    original = chordless_cycles(a)[0]
    reference_views = sorted(map(tuple, dihedral_views(a, original)))
    for shifted in [original[1:]+original[:1], original[::-1]]:
        assert sorted(map(tuple, dihedral_views(a, shifted))) == reference_views
    for _ in range(20):
        permuted = relabel(a, rng.permutation(8))
        invariance.append(float(np.max(abs(Sa-ring_summary(permuted, state, D, V, bridge)))))
    tree = Graph((0,)*4, ((0,1,0),(1,2,0),(2,3,0)))
    assert chordless_cycles(tree) == [] and np.array_equal(ring_summary(tree,state,D,V,bridge), np.zeros(97))
    union = Graph(a.atoms+b.atoms, a.edges+tuple((u+8,v+8,t) for u,v,t in b.edges))
    union_delta = float(np.max(abs(ring_summary(union,state,D,V,bridge)-Sa-Sb)))
    assert max(invariance + [union_delta]) < 1e-12
    # Chords, length boundary, disconnected graphs: catch enumeration mistakes.
    square_chord=Graph((0,)*4,((0,1,0),(0,2,0),(0,3,0),(1,2,0),(2,3,0)))
    assert sorted(map(len,chordless_cycles(square_chord)))==[3,3]
    assert sorted(map(len,chordless_cycles(closed_ring([0]*10))))==[10]
    assert chordless_cycles(closed_ring([0]*11))==[]
    report["invariance"] = dict(max_relabel_delta=max(invariance), dihedral_view_sets_equal=True,
                                disjoint_union_pool_delta=union_delta, empty_cycle_exact_zero=True)

    # Ring-only MAE signal through the shared D/V and ring MLP, finite difference.
    reader = rng.normal(size=97)
    def objective(s, dd, vv):
        pred = ring_summary(a,s,dd,vv,bridge) @ reader
        return abs(float(pred)-2.0)
    eps = 1e-5
    derivatives = {}
    for name in ["D", "V", "w1", "w2"]:
        array = D if name == "D" else V if name == "V" else state[name]
        direction = rng.normal(size=array.shape)
        direction /= np.linalg.norm(direction)
        s1, s2, d1, d2, v1, v2 = copy.deepcopy(state),copy.deepcopy(state),D.copy(),D.copy(),V.copy(),V.copy()
        plus = d1 if name == "D" else v1 if name == "V" else s1[name]
        minus = d2 if name == "D" else v2 if name == "V" else s2[name]
        plus += eps*direction
        minus -= eps*direction
        value = (objective(s1,d1,v1)-objective(s2,d2,v2))/(2*eps)
        assert np.isfinite(value) and abs(value) > 1e-8
        derivatives[name] = float(value)
    report["ring_only_MAE_directional_derivatives"] = derivatives
    ring_encoder_params = sum(x.size for x in state.values())
    reader_increment = 97*13
    assert RING_INPUT_DIM == 338 and ring_encoder_params == 24816
    report["parameter_design"] = dict(base=106925, ring_encoder=ring_encoder_params,
                                      reader_increment=reader_increment, total=106925+ring_encoder_params+reader_increment,
                                      ring_input=338, shared_dictionary_new_params=0)
    report["all_gates_passed"] = True
    return report


if __name__ == "__main__":
    report = run()
    Path(__file__).with_name("typed_cycle_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(report,indent=2,ensure_ascii=False))
