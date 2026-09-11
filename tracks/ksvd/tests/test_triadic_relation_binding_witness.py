"""Tests for the explicit triadic relation binding witness audit.

These pin the triad reconstruction, the permutation-invariance contract, the
unbound matched control and the adapter budgets used by
``experiments/luyin16/zinc_triadic_relation_binding_witness`` without touching
the real dataset, official valid/test or any frozen checkpoint unless the v4
export is already present.

1.  triple count equals C(n,3);
2.  every triple looks up its three pair states correctly (and endpoint order
    does not matter);
3.  triad_raw is invariant to pair-slot permutation;
4.  node relabelling does not change the molecule T_graph;
5.  sampling / enumeration is deterministic;
6.  the unbound control uses the same triple indices as the true witness;
7.  the unbound control preserves the pair-state marginal multiset;
8.  the unbound control actually breaks true triple association;
9.  the deterministic projection is identical across calls;
10. B2 (unbound) and E (true) have identical architecture;
11. B1 / B2 / E parameter budgets match within +-3%;
12. official valid/test are never loaded.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from tracks.ksvd.experiments.luyin16 import zinc_triadic_relation_binding_witness as tw


# ---------------------------------------------------------------------------
# synthetic export helpers
# ---------------------------------------------------------------------------

def _make_export(ns: list[int], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    pair_source_local: list[int] = []
    pair_target_local: list[int] = []
    pair_states: list[np.ndarray] = []
    pair_bucket: list[int] = []
    for n in ns:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        q = rng.standard_normal((len(pairs), tw.PAIR_DIM)).astype(np.float32)
        for (i, j) in pairs:
            pair_source_local.append(i)
            pair_target_local.append(j)
            pair_bucket.append(min(max(abs(i - j), 1), tw.N_BUCKETS) - 1)
        pair_states.append(q)
    n_patches = np.asarray(ns, dtype=np.int64)
    return {
        "subset_index": np.arange(len(ns), dtype=np.int64) + 1000,
        "n_patches": n_patches,
        "n_pairs": n_patches * (n_patches - 1) // 2,
        "pair_states": np.concatenate(pair_states, axis=0),
        "pair_bucket": np.asarray(pair_bucket, dtype=np.int64),
        "pair_source_local": np.asarray(pair_source_local, dtype=np.int64),
        "pair_target_local": np.asarray(pair_target_local, dtype=np.int64),
        "R": rng.standard_normal((len(ns), tw.R_DIM)).astype(np.float32),
        "target": rng.standard_normal(len(ns)),
        "yhat_0": rng.standard_normal(len(ns)),
    }


def _relabel_export(export: dict, permutation: dict[int, list[int]]) -> dict:
    """Relabel patch indices inside each molecule (same physical pairs)."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in export.items()}
    src = out["pair_source_local"].copy()
    tgt = out["pair_target_local"].copy()
    pair_off = np.concatenate([[0], np.cumsum(out["n_pairs"])]).astype(np.int64)
    for g in range(len(out["n_patches"])):
        q0, q1 = int(pair_off[g]), int(pair_off[g + 1])
        perm = np.asarray(permutation[g], dtype=np.int64)
        newsrc = perm[src[q0:q1]]
        newtgt = perm[tgt[q0:q1]]
        out["pair_source_local"][q0:q1] = np.minimum(newsrc, newtgt)
        out["pair_target_local"][q0:q1] = np.maximum(newsrc, newtgt)
    return out


# ---------------------------------------------------------------------------
# 1 -- triple count equals C(n,3)
# ---------------------------------------------------------------------------

def test_triple_count_equals_choose_n_3():
    ns = [3, 4, 6, 9, 12]
    export = _make_export(ns)
    witness = tw.TriadWitness(export)
    for p, n in enumerate(ns):
        assert witness.triple_count(p) == int(n * (n - 1) * (n - 2) // 6)
    report = tw._gate_triple_count(export, witness)
    assert report["passed"]


# ---------------------------------------------------------------------------
# 2 -- every triple looks up its three pair states correctly
# ---------------------------------------------------------------------------

def test_triple_lookup_integrity_and_endpoint_order():
    export = _make_export([5, 7])
    witness = tw.TriadWitness(export)
    for p in range(len(export["n_patches"])):
        q0 = int(witness.pair_offsets[p])
        q1 = int(witness.pair_offsets[p + 1])
        report = witness.molecules[p].lookup_integrity(witness.q[q0:q1])
        assert report["passed"]
        assert report["max_abs_diff"] == 0.0
        assert report["max_abs_diff_flipped"] == 0.0
    assert tw._gate_lookup_integrity(witness)["passed"]


# ---------------------------------------------------------------------------
# 3 -- triad_raw is invariant to pair-slot permutation
# ---------------------------------------------------------------------------

def test_triad_raw_permutation_invariance():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((20, tw.PAIR_DIM))
    b = rng.standard_normal((20, tw.PAIR_DIM))
    c = rng.standard_normal((20, tw.PAIR_DIM))
    base = tw.triad_raw_from_values(a, b, c)
    for perm in itertools.permutations([0, 1, 2]):
        vals = [a, b, c]
        other = tw.triad_raw_from_values(vals[perm[0]], vals[perm[1]], vals[perm[2]])
        assert np.array_equal(base, other)
    assert tw._gate_triad_invariance(tw.TriadWitness(_make_export([6, 8])))["passed"]


# ---------------------------------------------------------------------------
# 4 -- node relabelling does not change the molecule T_graph
# ---------------------------------------------------------------------------

def test_node_relabelling_invariance():
    ns = [6, 9]
    export = _make_export(ns, seed=7)
    witness = tw.TriadWitness(export)
    mean, scale = witness.fit_raw_statistics(np.arange(len(ns)))
    proj = tw._triad_projection()
    base = np.stack([
        witness.graph(p, unbound=False, mean=mean, scale=scale, projection=proj)
        for p in range(len(ns))
    ])
    permutation = {0: [3, 1, 5, 0, 2, 4], 1: [8, 0, 6, 3, 1, 7, 2, 5, 4]}
    relabelled = _relabel_export(export, permutation)
    witness2 = tw.TriadWitness(relabelled)
    # the fit statistics are computed over true triads and must be identical too
    mean2, scale2 = witness2.fit_raw_statistics(np.arange(len(ns)))
    assert np.allclose(mean, mean2, atol=tw.GRAPH_REBUILD_ATOL)
    assert np.allclose(scale, scale2, atol=tw.GRAPH_REBUILD_ATOL)
    other = np.stack([
        witness2.graph(p, unbound=False, mean=mean, scale=scale, projection=proj)
        for p in range(len(ns))
    ])
    assert np.allclose(base, other, atol=tw.GRAPH_REBUILD_ATOL)


# ---------------------------------------------------------------------------
# 5 -- enumeration / sampling is deterministic
# ---------------------------------------------------------------------------

def test_enumeration_and_sampling_deterministic():
    first = tw.enumerate_triples(8)
    second = tw.enumerate_triples(8)
    assert np.array_equal(first, second)
    assert first.shape == (56, 3)

    a = tw.sample_unordered_triples(30, cap=512, molecule_id="train:0001")
    b = tw.sample_unordered_triples(30, cap=512, molecule_id="train:0001")
    c = tw.sample_unordered_triples(30, cap=512, molecule_id="train:0002")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape[0] == 512
    # every sampled row is a valid unordered triple
    assert np.all(a[:, 0] < a[:, 1]) and np.all(a[:, 1] < a[:, 2])


# ---------------------------------------------------------------------------
# 6 -- unbound control uses the same triple indices as the true witness
# ---------------------------------------------------------------------------

def test_unbound_uses_same_triple_indices():
    export = _make_export([7, 10])
    witness = tw.TriadWitness(export)
    for p in range(len(export["n_patches"])):
        a_t, b_t, c_t = witness._states(p, unbound=False)
        a_u, b_u, c_u = witness._states(p, unbound=True)
        assert np.array_equal(a_t, a_u)
        assert np.array_equal(b_t, b_u)
        assert c_u.shape == c_t.shape
        # slots are a permutation of the true outer relations
        assert np.allclose(np.sort(c_u, axis=0), np.sort(c_t, axis=0))


# ---------------------------------------------------------------------------
# 7 -- unbound control preserves the pair-state marginal multiset
# ---------------------------------------------------------------------------

def test_unbound_preserves_pair_state_marginal_multiset():
    export = _make_export([8, 11])
    witness = tw.TriadWitness(export)
    for p in range(len(export["n_patches"])):
        a, b, c_t = witness._states(p, unbound=False)
        _, _, c_u = witness._states(p, unbound=True)
        true_all = np.concatenate([a, b, c_t], axis=0)
        unbound_all = np.concatenate([a, b, c_u], axis=0)
        # exact multiset identity: same rows, same multiplicities
        true_sorted = true_all[np.lexsort(true_all.T[::-1])]
        unbound_sorted = unbound_all[np.lexsort(unbound_all.T[::-1])]
        assert np.array_equal(true_sorted, unbound_sorted)


# ---------------------------------------------------------------------------
# 8 -- unbound control actually breaks true triple association
# ---------------------------------------------------------------------------

def test_unbound_breaks_true_triple_association():
    export = _make_export([9, 12, 15], seed=11)
    witness = tw.TriadWitness(export)
    total = 0
    fixed = 0
    same_row = 0
    for p in range(len(export["n_patches"])):
        mol = witness.molecules[p]
        q0 = int(witness.pair_offsets[p])
        q1 = int(witness.pair_offsets[p + 1])
        perm, info = tw.unbound_permutation(
            mol.outer_buckets(witness.pair_bucket[q0:q1]),
            witness.molecule_id(p),
            outer_rows=mol.c,
        )
        total += info["n_triples"]
        fixed += info["n_fixed_points"]
        same_row += info["n_same_outer_row"]
    assert total > 0
    assert fixed / total < 0.05
    # the dominant association (which outer relation closes which a,b pair) is broken
    assert 1.0 - same_row / total > 0.85


# ---------------------------------------------------------------------------
# 9 -- deterministic projection
# ---------------------------------------------------------------------------

def test_projection_deterministic_and_orthonormal():
    p1 = tw._triad_projection()
    p2 = tw._triad_projection()
    assert np.array_equal(p1, p2)
    assert p1.shape == (tw.TRIAD_RAW_DIM, tw.TRIAD_PROJ_DIM)
    assert np.allclose(p1.T @ p1, np.eye(tw.TRIAD_PROJ_DIM), atol=1e-12)


# ---------------------------------------------------------------------------
# 10 -- B2 (unbound) and E (true) have identical architecture
# ---------------------------------------------------------------------------

def test_unbound_and_true_architectures_identical():
    b2 = tw._make_head(0, "unbound")
    e = tw._make_head(0, "true")
    sd2 = b2.state_dict()
    sde = e.state_dict()
    assert list(sd2.keys()) == list(sde.keys())
    for key in sd2:
        assert sd2[key].shape == sde[key].shape
    assert tw._count_parameters(b2) == tw._count_parameters(e)


# ---------------------------------------------------------------------------
# 11 -- B1 / B2 / E parameter budgets match within +-3%
# ---------------------------------------------------------------------------

def test_parameter_budgets_match():
    b1 = tw._count_parameters(tw.ResidualHead(tw.R_DIM, tw.RONLY_HIDDEN, tw.RONLY_DEPTH))
    b2 = tw._count_parameters(
        tw.ResidualHead(tw.R_DIM + tw.T_GRAPH_DIM, tw.WITNESS_HIDDEN, tw.WITNESS_DEPTH)
    )
    assert b1 == 4135
    assert b2 == 4189
    assert abs(b2 - b1) / b1 <= 0.03
    assert tw._gate_parameter_match()["passed"]


# ---------------------------------------------------------------------------
# 12 -- official valid/test are never loaded
# ---------------------------------------------------------------------------

def test_official_valid_test_never_loaded():
    source = Path(tw.__file__).read_text(encoding="utf-8")
    docstring = tw.__doc__ or ""
    body = source.replace(docstring, "")
    forbidden = [
        "official_valid",
        "official_test",
        "valid_loader",
        "test_loader",
        "_load_valid",
        "_load_test",
        "valid_records",
        "test_records",
    ]
    for token in forbidden:
        assert token not in body, f"forbidden official-split reference: {token}"

    # the split manifest only ever draws from the official train universe
    manifest = tw._fold_split_manifest(0)
    for role in ("adapter_fit", "adapter_selection", "adapter_evaluation"):
        assert all(mid.startswith("train:") for mid in manifest["molecule_ids"][role])
    assert manifest["sizes"] == {
        "adapter_fit": 1200,
        "adapter_selection": 400,
        "adapter_evaluation": 400,
    }


# ---------------------------------------------------------------------------
# extra: real v4 export contract (skipped when the frozen export is absent)
# ---------------------------------------------------------------------------

V4_PATH = tw._export_path(0, 0)


@pytest.mark.skipif(not V4_PATH.exists(), reason="frozen v4 export not present")
def test_real_v4_export_triple_gate():
    export = tw.load_v4_export(V4_PATH)
    assert export["fingerprint"]["export_version"] == tw.V4_EXPORT_VERSION
    witness = tw.TriadWitness(export)
    report = tw._gate_triple_count(export, witness)
    assert report["passed"]
    assert tw._gate_pair_integrity(export)["passed"]
    assert tw._gate_lookup_integrity(witness)["passed"]
