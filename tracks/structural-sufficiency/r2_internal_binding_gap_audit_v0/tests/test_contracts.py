"""Fast contract checks for the read-only R2 internal-binding audit."""

from __future__ import annotations

import numpy as np

import ib_common as C


def test_expected_identities() -> None:
    assert C.EXPECTED_IDENTITY["D0"].startswith("60ac13c0")
    assert C.EXPECTED_IDENTITY["D1"].startswith("c94e2d4b")
    assert C.EXPECTED_IDENTITY["D0"] != C.EXPECTED_IDENTITY["D1"]


def test_ego_packing_and_edge_filters() -> None:
    import sr_gnn as SG
    import sr_struct as SS

    bundle = C.load_bundle("D0")
    row = int(bundle.eval_rows[0])
    st = SS.build_structures(bundle, [row])[row]
    SG.prepare_ranges(st)
    ref = SG.ego_block(st, 2)
    block0, stats0 = C.ego_block_with_stats(st, 2, None)
    for key in ("atom", "src", "dst", "bond", "ptr"):
        assert np.array_equal(block0[key], ref[key])
    for drop in ("11", "22"):
        block, stats = C.ego_block_with_stats(st, 2, drop)
        assert np.array_equal(block["atom"], ref["atom"])
        assert np.array_equal(block["ptr"], ref["ptr"])
        assert np.array_equal(stats["total_per_root"], stats0["total_per_root"])
        assert np.all(stats["removed_per_root"] <= stats["total_per_root"])
        assert int(stats["removed_per_root"].sum()) <= int(
            stats0["total_per_root"].sum())
    # A molecule with no d2--d2 edge must be byte-identical under drop22.
    clean_row = None
    for candidate in bundle.eval_rows[:100].tolist():
        stc = SS.build_structures(bundle, [int(candidate)])[int(candidate)]
        _, stc_stats = C.ego_block_with_stats(stc, 2, "22")
        if int(stc_stats["removed_per_root"].sum()) == 0:
            clean_row = int(candidate)
            break
    assert clean_row is not None
    stc = SS.build_structures(bundle, [clean_row])[clean_row]
    refc = SG.ego_block(stc, 2)
    blockc, _ = C.ego_block_with_stats(stc, 2, "22")
    for key in ("atom", "src", "dst", "bond", "ptr"):
        assert np.array_equal(blockc[key], refc[key])


def test_topology_features_are_nonnegative() -> None:
    import gwic_data as GD

    bundle = C.load_bundle("D0")
    view = GD.molecule_view(bundle.raw, int(bundle.eval_rows[0]))
    rec = C.r2_topology(view)
    assert rec["n_roots"] == rec["n_atoms"]
    for key in ("closure_11_mean", "closure_22_mean", "reconvergence_2_mean",
                "n_nodes_mean", "n_edges_mean", "cycle_rank_mean",
                "supports3_mean", "supports4_mean"):
        assert np.isfinite(rec[key])
        assert rec[key] >= 0.0
    assert 0.0 <= rec["ring_frac"] <= 1.0


def test_bootstrap_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    pa = y + rng.normal(scale=0.5, size=40)
    pb = y + rng.normal(scale=0.4, size=40)
    clusters = np.asarray([f"c{i // 4}" for i in range(40)], dtype=object)
    a = C.bootstrap_paired(y, pa, pb, clusters, n_boot=200, seed=1703)
    b = C.bootstrap_paired(y, pa, pb, clusters, n_boot=200, seed=1703)
    assert a["gain_mae"] == b["gain_mae"]
    assert a["gain_mae_ci"] == b["gain_mae_ci"]
    assert np.array_equal(a["draws"]["gain_mae"], b["draws"]["gain_mae"])