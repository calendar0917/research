"""Model-side correctness: tree executor, block trick, label isolation, reload."""

from __future__ import annotations

import numpy as np
import pytest

import rd_arms as A
import rd_common as C
import rd_eval as E
import rd_pairs as P
import rd_refs as R
import rd_tree as T
from helpers import make_task


def _context(task):
    return A.ArmContext.build(task, R.reference_table(task))


def _tiny_xgb(matrix, target, *, rounds=6, depth=3, seed=0):
    import xgboost as xgb

    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cpu",
        "max_depth": depth,
        "eta": 0.3,
        "min_child_weight": 1,
        "reg_lambda": 0.0,
        "max_bin": 16,
        "base_score": 0.0,
        "seed": seed,
        "nthread": 2,
    }
    return xgb.train(params, xgb.DMatrix(matrix, label=target), num_boost_round=rounds)


def test_block_layout_is_the_transposed_pair_matrix():
    task = make_task(n_train=12, n_valid=3, n_test=3, n_bits=6, n_descriptors=1)
    order = P.eligible_order_train(task)
    spec = P.build_pairs(task, order)
    src = spec["src_local"][:300].astype(np.int64)
    ref = spec["ref_local"][:300].astype(np.int64)
    n_rows = int(src.size)
    forward = P.pair_matrix(task, src, ref)
    reverse = P.pair_matrix(task, ref, src)
    blocks = P.pair_feature_blocks(task, src, ref)
    booster = _tiny_xgb(forward, spec["target_std"][:300])
    trees = T.extract_trees(booster, [f"c{i}" for i in range(forward.shape[1])], rounds=6)

    dense_fwd = np.zeros(n_rows, dtype=np.float32)
    dense_rev = np.zeros(n_rows, dtype=np.float32)
    T.accumulate(dense_fwd, trees, T.DenseLayout(forward), stop=6)
    T.accumulate(dense_rev, trees, T.DenseLayout(reverse), stop=6)

    block_fwd = np.zeros(n_rows, dtype=np.float32)
    block_rev = np.zeros(n_rows, dtype=np.float32)
    T.accumulate(block_fwd, trees, T.BlockLayout(blocks, task.n_features, reverse=False), stop=6)
    T.accumulate(block_rev, trees, T.BlockLayout(blocks, task.n_features, reverse=True), stop=6)

    assert np.array_equal(block_fwd, dense_fwd)
    assert np.array_equal(block_rev, dense_rev)

    import xgboost as xgb

    xgb_fwd = booster.predict(xgb.DMatrix(forward), iteration_range=(0, 6))
    xgb_rev = booster.predict(xgb.DMatrix(reverse), iteration_range=(0, 6))
    assert np.max(np.abs(np.asarray(xgb_fwd) - block_fwd)) <= 1e-6
    assert np.max(np.abs(np.asarray(xgb_rev) - block_rev)) <= 1e-6


def test_d_arm_records_every_round_and_matches_xgboost():
    task = make_task(n_train=44, n_valid=10, n_test=10, n_bits=8, n_descriptors=2)
    ctx = _context(task)
    run = A.run_d(ctx, preset="H1", device="cpu", rounds=6)
    assert run.trained_rounds == 6
    assert len(run.curves) == 6 * 3 * (1 + len(C.CONFIGS))
    index = run.curve_index()
    report = A.verify_d(
        ctx,
        booster=run.booster,
        trees=run.trees,
        round_index=6,
        recorded_predictions=run.final_predictions,
        curves_index=index,
    )
    for split in C.SPLITS:
        entry = report["splits"][split]
        assert entry["max_abs_sample_diff_raw"] <= 1e-6
        assert entry["rmse_abs_diff"] <= 1e-6
    # the R state is exactly the D state plus the reference correction
    chosen = C.CONFIGS[0]["name"]
    manual = A.d_r_predictions_at(ctx, run.final_predictions, chosen, "valid")
    assert np.allclose(manual, E.aggregate_r(
        run.final_predictions["valid"],
        np.concatenate([run.final_predictions[s] for s in C.SPLITS])[
            ctx.ref_table["splits"]["valid"]["ref_local"]
        ],
        ctx.weights["valid"][chosen],
        ctx.y_ref["valid"],
    ), rtol=0.0, atol=0.0)


def test_val_test_labels_cannot_move_the_training_trajectory():
    task = make_task(n_train=44, n_valid=10, n_test=10)
    ctx = _context(task)
    run_a = A.run_d(ctx, preset="H1", device="cpu", rounds=5)

    shuffled = make_task(n_train=44, n_valid=10, n_test=10)
    shuffled.y["valid"] = shuffled.y["valid"][::-1].copy()
    shuffled.y["test"] = shuffled.y["test"][::-1].copy()
    ctx_b = _context(shuffled)
    run_b = A.run_d(ctx_b, preset="H1", device="cpu", rounds=5)

    idx_a = run_a.curve_index()
    idx_b = run_b.curve_index()
    for round_index in range(1, 6):
        assert idx_a[(round_index, "train", "none")]["rmse_raw"] == pytest.approx(
            idx_b[(round_index, "train", "none")]["rmse_raw"], rel=0, abs=0.0
        )
    for split in ("valid", "test"):
        assert idx_a[(5, split, "none")]["rmse_raw"] != pytest.approx(
            idx_b[(5, split, "none")]["rmse_raw"], rel=0, abs=1e-12
        )
    import xgboost as xgb

    model_a = bytes(run_a.booster.save_raw(raw_format="json"))
    model_b = bytes(run_b.booster.save_raw(raw_format="json"))
    assert model_a == model_b
    # and the reloaded booster reproduces the selected prediction exactly
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    run_a.booster.save_model(str(tmp / "d.ubj"))
    reloaded = xgb.Booster(model_file=str(tmp / "d.ubj"))
    names = list(task.feature_names)
    pred = reloaded.predict(xgb.DMatrix(task.x["valid"], feature_names=names))
    assert np.allclose(
        ctx.mu + ctx.sigma * np.asarray(pred), run_a.final_predictions["valid"], rtol=0, atol=1e-6
    )


def test_c_arm_algebra_and_full_prefix_recompute():
    task = make_task(n_train=40, n_valid=6, n_test=6, n_bits=6, n_descriptors=1)
    ctx = _context(task)
    order = P.eligible_order_train(task)
    spec = P.build_pairs(task, order)
    run = A.run_c(ctx, spec, preset="H1", device="cpu", rounds=4)
    assert run.trained_rounds == 4
    assert len(run.curves) == 4 * 3 * len(C.CONFIGS)
    assert len(run.pair_curves) == 4
    config = C.CONFIGS[2]["name"]
    state = A.c_state(ctx, booster=run.booster, trees=run.trees, rounds=3, config_name=config)
    for split in C.SPLITS:
        pred = state["pred"][split]
        assert pred.shape == (task.n(split),)
        # C equals sum_j w_j (y_j + delta) recomputed from the saved delta
        weights = ctx.weights[split][config]
        recon = np.sum(
            np.asarray(weights, dtype=np.float64)
            * (ctx.y_ref[split] + state["delta"][split].astype(np.float64)),
            axis=1,
        )
        assert np.allclose(pred, recon, rtol=0.0, atol=1e-5)
    index = {(row["round"], row["split"], row["config"]): row for row in run.curves}
    sample_rows = {
        split: A.sample_flat_rows(int(task.n(split) * C.MAX_REFERENCES), 32) for split in C.SPLITS
    }
    state_checked = A.c_state(
        ctx,
        booster=run.booster,
        trees=run.trees,
        rounds=3,
        config_name=config,
        sample_rows=sample_rows,
    )
    report = A.verify_c(
        ctx,
        booster=run.booster,
        pair_spec=spec,
        final_margins=run.final_margins,
        state=state_checked,
        round_index=3,
        final_round=4,
        curves_index=index,
        config_name=config,
        sample_count=64,
    )
    for split in C.SPLITS:
        entry = report["splits"][split]
        assert entry["rmse_abs_diff"] <= 1e-12
        assert entry["max_abs_margin_diff_sel_fwd"] <= 1e-6
        assert entry["max_abs_margin_diff_sel_rev"] <= 1e-6
    assert report["pair_sample"]["max_abs_margin_diff_final_fwd"] <= 1e-6
    assert report["pair_sample"]["max_abs_margin_diff_final_rev"] <= 1e-6


def test_selection_tie_breaks_follow_the_declared_order():
    from types import SimpleNamespace

    def run(preset, rows):
        curves = []
        for round_index, value, config in rows:
            curves.append(
                {
                    "dataset": "x",
                    "method": "C",
                    "preset": preset,
                    "round": round_index,
                    "split": "valid",
                    "config": config,
                    "rmse_raw": value,
                    "mae_raw": value,
                }
            )
        return SimpleNamespace(preset=preset, curves=curves)

    results = [
        run("H2", [(5, 0.5, "K4_uniform")]),
        run("H1", [(9, 0.5, "K64_softmax_tau_0.1"), (9, 0.5, "K4_uniform")]),
    ]
    chosen = A.select_c(results)
    assert chosen["preset"] == "H1"
    assert chosen["round"] == 9
    assert chosen["config"] == "K4_uniform"

    n_states = [
        {"config": "K64_uniform", "valid_rmse_raw": 0.5},
        {"config": "K16_softmax_tau_0.1", "valid_rmse_raw": 0.5},
        {"config": "K16_uniform", "valid_rmse_raw": 0.5},
    ]
    assert A.select_config(n_states)["config"] == "K16_uniform"
