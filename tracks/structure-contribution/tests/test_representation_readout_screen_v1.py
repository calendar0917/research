"""cscl-strong-representation-readout-screen-v1 correctness tests.

Fast tests (no data / no checkpoints): Typed-WL hash/descriptor invariants,
count conservation, radius-salted hash spaces, protocol freeze and the frozen
XGBoost recipe cross-check.

Slow tests (marked ``slow``): build Arm-A features for a few real ZINC rows and
cross-check the label-free global structure against the historical standardized
cache inversion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
for _p in (str(_CODE_DIR),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import typed_wl_features as tw  # noqa: E402

slow = pytest.mark.slow

CONFIG_PATH = REPO_ROOT / "tracks/structure-contribution/configs/strong_representation_readout_screen_v1.yaml"


# ---------------------------------------------------------------------------
# fast: Typed-WL primitives
# ---------------------------------------------------------------------------


def test_descriptor_dimensions_and_conservation():
    # path C-C-C with three bond types
    d = tw.wl_descriptor([0, 0, 0], [(0, 1, 0), (1, 2, 1)])
    assert d.shape == (tw.WL_DIM,)
    assert np.isfinite(d).all() and (d >= 0).all()
    per_radius = d.reshape(len(tw.WL_RADII), tw.BUCKETS_PER_RADIUS).sum(axis=1)
    assert np.array_equal(per_radius, np.array([3.0, 3.0, 3.0, 3.0]))
    # single atom: one rooted fragment per radius
    one = tw.wl_descriptor([5], [])
    assert one.reshape(4, tw.BUCKETS_PER_RADIUS).sum(axis=1).tolist() == [1.0, 1.0, 1.0, 1.0]
    assert float(one.sum()) == 4.0


def test_hash_is_sha256_stable_and_radius_salted():
    # deterministic across calls (never Python hash())
    assert tw.bucket_of(0, "abc") == tw.bucket_of(0, "abc")
    assert tw.bucket_of(2, "abc") == tw.bucket_of(2, "abc")
    assert tw.stable_sha256_int("x") == tw.stable_sha256_int("x")
    # radius is part of the bucket payload -> independent hash spaces
    shared = sum(1 for c in [f"c{i}" for i in range(200)] if len({tw.bucket_of(r, c) for r in tw.WL_RADII}) == 1)
    assert shared == 0
    for r in tw.WL_RADII:
        assert 0 <= tw.bucket_of(r, "abc") < tw.BUCKETS_PER_RADIUS


def test_color_update_uses_sorted_multiset():
    # colours must not depend on adjacency insertion order
    adj_a = [[(0, 1), (1, 2)], [(0, 0)], [(1, 0)]]
    adj_b = [[(1, 2), (0, 1)], [(0, 0)], [(1, 0)]]
    ca = tw.color_round(1, ["a0", "a1", "a2"], adj_a)
    cb = tw.color_round(1, ["a0", "a1", "a2"], adj_b)
    assert ca == cb


def test_atom_order_permutation_invariance():
    rng = np.random.default_rng(7)
    atom_types = [0, 1, 2, 0, 3, 1]
    bonds = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 4, 2), (4, 5, 0), (5, 0, 1), (1, 4, 0)]
    base = tw.wl_descriptor(atom_types, bonds)
    for _ in range(5):
        perm = rng.permutation(len(atom_types))
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        new_types = [atom_types[i] for i in perm]
        new_bonds = sorted(
            (int(min(inv[u], inv[v])), int(max(inv[u], inv[v])), int(bt)) for u, v, bt in bonds
        )
        if new_bonds != bonds:
            assert np.array_equal(base, tw.wl_descriptor(new_types, new_bonds))


def test_bidirectional_edges_are_deduplicated():
    x = np.array([0, 1, 0], dtype=np.int64)
    bonds = [(0, 1, 0), (1, 2, 1)]
    src, dst, typ = [], [], []
    for u, v, bt in bonds:
        src += [u, v]
        dst += [v, u]
        typ += [bt, bt]
    pyg_like = tw.molecule_typed_wl_from_arrays(
        x, np.asarray([src, dst], dtype=np.int64), np.asarray(typ, dtype=np.int64)
    )
    unique = tw.molecule_typed_wl_from_arrays(
        x,
        np.asarray([[0, 1], [1, 2]], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )
    assert np.array_equal(pyg_like, unique)


def test_bond_type_exchange_changes_descriptor():
    x = np.array([0, 0, 0], dtype=np.int64)
    ei = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    a = tw.molecule_typed_wl_from_arrays(x, ei, np.asarray([0, 0], dtype=np.int64))
    b = tw.molecule_typed_wl_from_arrays(x, ei, np.asarray([1, 0], dtype=np.int64))
    assert not np.array_equal(a, b)


def test_inconsistent_directed_bond_types_rejected():
    with pytest.raises(ValueError):
        tw.unique_undirected_bonds(
            np.asarray([[0, 1], [1, 0]], dtype=np.int64),
            np.asarray([0, 1], dtype=np.int64),
        )


def test_collision_stats_shape():
    # realistic shape: one entry per molecule, each a list over radii of per-atom colours
    colors = [[["a0", "a1"], ["b0", "b1"]], [["a0", "a2"], ["b0", "b2"]], [["a0", "a1"], ["b0", "b1"]]]
    stats = tw.bucket_collision_stats(colors)
    assert set(stats) == {f"r{r}" for r in tw.WL_RADII}
    for radius in tw.WL_RADII:
        assert stats[f"r{radius}"]["distinct_signatures"] >= stats[f"r{radius}"]["occupied_buckets"]


# ---------------------------------------------------------------------------
# fast: frozen protocol / recipe
# ---------------------------------------------------------------------------


def test_protocol_config_frozen_constants():
    import yaml

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["protocol_id"] == "cscl-strong-representation-readout-screen-v1"
    assert config["seeds_allowed"] == [0]
    assert config["arm_A_typed_wl"]["radii"] == [0, 1, 2, 3]
    assert config["arm_A_typed_wl"]["buckets_per_radius"] == 1024
    assert config["arm_A_typed_wl"]["wl_dim"] == 4096
    assert config["arm_A_typed_wl"]["total_dim"] == 4151
    assert config["arm_B_frozen_reader_input"]["expected_dim"] == 814
    assert config["judgment_rules_preefrozen"]["monitor_signal_threshold"] == 0.10826


def test_frozen_xgb_recipe_matches_historical_defaults():
    import run_representation_readout_screen_v1 as drv
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import DEFAULT_XGB_PARAMS

    for key, value in DEFAULT_XGB_PARAMS.items():
        assert drv.FROZEN_XGB_PARAMS[key] == value, key
    assert drv.FROZEN_XGB_PARAMS["objective"] == "reg:squarederror"
    assert drv.FROZEN_XGB_PARAMS["eval_metric"] == "mae"
    assert drv.FROZEN_XGB_PARAMS["tree_method"] == "hist"
    assert drv.FROZEN_XGB_PARAMS["random_state"] == 0
    assert drv.FROZEN_XGB_PARAMS["n_jobs"] == 8
    assert drv.FROZEN_XGB_PARAMS["n_estimators"] == 500


def test_driver_never_references_forbidden_sources():
    source = Path(_CODE_DIR / "run_representation_readout_screen_v1.py").read_text(encoding="utf-8")
    for token in ("encoded_valid", "encoded_test", "official_valid.pt", "official_test.pt", "target_decomposition"):
        assert token not in source, token


def test_arm_dimensions_frozen():
    import run_representation_readout_screen_v1 as drv

    assert drv.ARM_A_DIM == 4151
    assert drv.ARM_B_DIM == 814
    assert tw.WL_DIM == 4096


def test_paired_bootstrap_direction_and_ci():
    import run_representation_readout_screen_v1 as drv

    rng = np.random.default_rng(0)
    err_b = rng.random(500)
    err_a = err_b - 0.05  # A strictly better by 0.05
    out = drv.paired_bootstrap(err_a, err_b, resamples=200, seed=0)
    assert out["mean_diff_signed"] == pytest.approx(-0.05, abs=1e-12)
    assert out["ci95"][1] < 0.0
    assert out["frac_a_better"] == 1.0


# ---------------------------------------------------------------------------
# slow: data-backed Arm-A build checks
# ---------------------------------------------------------------------------


@slow
def test_arm_a_real_rows_counts_and_global_cross_check():
    import torch

    import run_representation_readout_screen_v1 as drv
    from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztf

    dataset = drv.load_dataset()
    _smiles, idx = drv.split_indices()
    drv.row_hash_check(idx, log=lambda *a, **k: None)
    blob10 = {k: z for k, z in np.load(drv.LEGACY_PREP, allow_pickle=False).items()}
    ctx_std = drv._legacy_std(blob10, "ctx")
    encoded = torch.load(
        Path(drv.cf.REPO_ROOT) / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt",
        map_location="cpu",
        weights_only=False,
    )
    frame = ztf.load_cache("train")
    check_rows = [0, 1, 2, 941, 2264, 4100, 7500, 9999]
    max_global = 0.0
    for i in check_rows:
        data = dataset[i]
        x = data.x.numpy().reshape(-1).astype(np.int64)
        ei = data.edge_index.numpy().astype(np.int64)
        ea = data.edge_attr.numpy().reshape(-1).astype(np.int64)
        bonds = tw.unique_undirected_bonds(ei, ea)
        d = tw.wl_descriptor(x, bonds)
        assert np.array_equal(d.reshape(4, 1024).sum(1), np.full(4, len(x)))
        g = tw.global_structure_raw(tw.graph_from_data(data))
        raw_cache = ctx_std.inverse(encoded[i].global_context.numpy().reshape(-1))[:30]
        max_global = max(max_global, float(np.abs(g - raw_cache).max()))
        tv = ztf.raw_vector(frame.iloc[i], "hinge")
        raw_topo = drv._legacy_std(blob10, "topo").inverse(encoded[i].topology_features.numpy().reshape(-1))
        assert float(np.abs(tv - raw_topo).max()) <= 1e-4  # historical float32 round-trip max 6.1e-5
    assert max_global <= 1e-5
