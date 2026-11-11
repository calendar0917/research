"""Unit tests for the efficiency + width round (CPU only, no formal GPU run).

Covers the frozen recipe/parameter identities, the equivalence of every
optimised component against the historical code path, and the prefetch order.
The real 1M-cache tests are skipped when the cache is absent; the smoke cache
that the historical round already produced is used for the training-step tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ew_common as C  # noqa: E402
import ew_collate as EC  # noqa: E402
import ew_data as DATA  # noqa: E402
import ew_model as M  # noqa: E402
import ew_pipeline as PIPE  # noqa: E402
import ew_train as T  # noqa: E402

torch = pytest.importorskip("torch")

REAL_CACHE = C.OLD_CACHE / "pcqm_train1m"
SMOKE_TRAIN = C.OLD_CACHE / "pcqm_smoke_train"
SMOKE_VALID = C.OLD_CACHE / "pcqm_smoke_valid"


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} is not present on this host")


# ---------------------------------------------------------------------------
# frozen identities
# ---------------------------------------------------------------------------


def test_recipe_is_preregistered_and_unchanged():
    C.assert_recipe()
    assert C.BATCH_SIZE == 128
    assert C.TOTAL_EPOCHS == 40
    assert C.STEPS_PER_EPOCH == 7_813 and C.TOTAL_STEPS == 312_520
    assert C.MODEL_SEED == 0 and C.ORDER_SEED == 0
    assert set(C.RUNS_TABLE) == {"PCQM_W1_1M", "PCQM_W3_1M"}
    assert C.RUNS_TABLE["PCQM_W1_1M"]["scale"] == 1
    assert C.RUNS_TABLE["PCQM_W3_1M"]["scale"] == 3
    assert all(v["train_size"] == 1_000_000 and v["epochs"] == 40 for v in C.RUNS_TABLE.values())
    assert C.HISTORICAL["run"] == "PCQM_W2_1M"
    assert C.HISTORICAL["parameters"] == 939_079
    assert abs(C.HISTORICAL["best_valid_mae_eV"] - 0.132310904) < 1e-9


def test_widths_are_the_inherited_table_times_scale():
    reference = C.PQ.BS.config_for_scale(1)
    for scale in (1, 2, 3):
        cfg = C.WIDTHS[scale]
        assert cfg["n_out"] == 1
        assert cfg["n_heads"] == 4
        for key in C.PQ.REFERENCE_WIDTHS:
            assert cfg[key] == scale * reference[key]
    # W2 is exactly the historical width table
    for key, value in C.PQ.BS.config_for_scale(2).items():
        if key == "n_out":
            assert value == 128
            continue
        assert C.WIDTHS[2][key] == value


def test_analytic_parameter_counts_match_preregistration():
    table = C.param_table()
    assert table["1"]["total_parameters"] == 259_831
    assert table["2"]["total_parameters"] == 939_079
    assert table["3"]["total_parameters"] == 2_038_167
    for entry in table.values():
        assert entry["matches_expected"]
        assert entry["total_parameters"] == entry["base_parameters"] + entry["readout_parameters"]
    # W2 = the historical pilot's parameter count
    assert table["2"]["total_parameters"] == C.HISTORICAL["parameters"]


def test_lr_schedule_is_the_frozen_40_epoch_one():
    audit = C.PQ.lr_audit(C.TRAIN_ROWS)
    assert audit["steps_per_epoch"] == 7_813
    assert audit["total_steps_40_epochs"] == 312_520
    assert abs(audit["lr_step0"] - (3e-5 + (3e-4 - 3e-5) / 7_813)) < 1e-15
    assert abs(audit["lr_last_step_used"] - 1e-5) < 1e-9
    # the formal horizon is 40 epochs, never a short trial horizon
    assert C.scheduled_lr(312_519, steps_epoch=7_813, steps_total=312_520) >= 1e-5


def test_frozen_w2_builder_is_bitwise_identical():
    import feature_spec as FS

    spec = FS.lipo_v1_spec()
    norm = {
        "global_mean": np.zeros(spec.global_dim, dtype=np.float32),
        "global_scale": np.ones(spec.global_dim, dtype=np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, dtype=np.float32),
        "scalar_scale": np.ones(spec.n_scalars, dtype=np.float32),
    }
    eq = M.historical_w2_equivalence(spec, norm, seed=0)
    assert eq["bitwise_identical"], eq
    assert eq["max_abs_delta"] == 0.0
    assert eq["parameters_a"] == eq["parameters_b"] == 939_079


def test_scale_builders_have_the_registered_parameter_counts():
    import feature_spec as FS

    spec = FS.lipo_v1_spec()
    norm = {
        "global_mean": np.zeros(spec.global_dim, dtype=np.float32),
        "global_scale": np.ones(spec.global_dim, dtype=np.float32),
        "scalar_mean": np.zeros(spec.n_scalars, dtype=np.float32),
        "scalar_scale": np.ones(spec.n_scalars, dtype=np.float32),
    }
    for scale in (1, 3):
        with torch.random.fork_rng(devices=[]):
            model, audit = M.build_model(spec, norm, scale=scale, seed=0, device="cpu")
        assert int(audit["total_parameters"]) == C.EXPECTED_PARAMETERS[scale]
        assert audit["width_audit"]["all_match"]
        assert audit["readout_output_shape"] == [1, C.WIDTHS[scale]["pool_hidden"]]


# ---------------------------------------------------------------------------
# E1: collation equality
# ---------------------------------------------------------------------------


def _smoke_task(*, epochs: int = 3, scale: int = 1) -> dict:
    import feature_spec as FS

    spec = FS.load_spec(C.OLD_ANALYSIS / "pcqm_spec.json")
    feature_norm = C.PQ.read_json(C.OLD_ANALYSIS / "pcqm_feature_norm.json")
    label_norm_payload = C.PQ.read_json(C.OLD_ANALYSIS / "pcqm_label_norm.json")
    train_cache = C.PQ_DATA.load_cache(SMOKE_TRAIN)
    valid_cache = C.PQ_DATA.load_cache(SMOKE_VALID)
    labels = np.load(C.OLD_CACHE / "pcqm_labels_trainvalid.npy").astype(np.float64)
    train_gids = np.asarray(train_cache["gids"], dtype=np.int64)
    valid_gids = np.asarray(valid_cache["gids"], dtype=np.int64)
    train_y_eV = labels[train_gids]
    valid_y_eV = labels[valid_gids]
    norm = {
        "global_mean": np.asarray(feature_norm["global_mean_list"], dtype=np.float32),
        "global_scale": np.asarray(feature_norm["global_scale_list"], dtype=np.float32),
        "scalar_mean": np.asarray(feature_norm["scalar_mean_list"], dtype=np.float32),
        "scalar_scale": np.asarray(feature_norm["scalar_scale_list"], dtype=np.float32),
    }
    n_rows = int(train_cache["n_molecules"])
    return {
        "run": "PCQM_W1_1M",
        "model": f"W{scale}",
        "model_name": C.MODEL_NAMES[scale],
        "scale": int(scale),
        "expected_parameters": C.EXPECTED_PARAMETERS[scale],
        "widths": C.WIDTHS[scale],
        "spec": spec,
        "norm": norm,
        "feature_norm": feature_norm,
        "label_norm": label_norm_payload,
        "train_cache": train_cache,
        "valid_cache": valid_cache,
        "valid_rows": np.arange(valid_gids.size, dtype=np.int64),
        "valid_y_eV": valid_y_eV,
        "train_y_eV": train_y_eV,
        "train_y_z": C.PQ_DATA.to_z(train_y_eV, label_norm_payload).astype(np.float32),
        "monitor_rows": np.arange(min(256, n_rows), dtype=np.int64),
        "monitor_y_eV": train_y_eV[: min(256, n_rows)],
        "monitor_plan": {"n": int(min(256, n_rows)), "source": "test"},
        "n_rows": n_rows,
        "steps_epoch": int(np.ceil(n_rows / C.BATCH_SIZE)),
        "steps_total": int(C.TOTAL_EPOCHS * np.ceil(n_rows / C.BATCH_SIZE)),
        "seed": 0,
        "orders": C.PQ_DATA.epoch_orders(n_rows, seed=0, epochs=epochs),
        "evaluator": C.PQ_DATA._evaluator(),
        "cache_manifest": {"smoke": True},
        "data_isolation": {"smoke": True},
        "train_cache_rows": n_rows,
    }


def test_collate_v2_is_field_identical_on_real_cache():
    _require(REAL_CACHE)
    cache = C.PQ_DATA.load_cache(REAL_CACHE)
    rng = np.random.default_rng(11)
    for batch_size in (1, 2, 3, 128):
        rows = rng.permutation(int(cache["n_molecules"]))[:batch_size].tolist()
        a = C.PQ_DATA.collate(cache, rows)
        b = EC.collate_v2(cache, rows)
        cmp = EC.compare_batches(a, b)
        assert cmp["all_equal"], {k: v for k, v in cmp["fields"].items() if not v["equal"]}
    # all fields are present and typed exactly as the frozen collation
    assert sorted(a) == sorted(b)
    for key in a:
        if torch.is_tensor(a[key]):
            assert a[key].dtype == b[key].dtype
            assert tuple(a[key].shape) == tuple(b[key].shape)


def test_collate_v2_rejects_bad_rows():
    _require(REAL_CACHE)
    cache = C.PQ_DATA.load_cache(REAL_CACHE)
    n = int(cache["n_molecules"])
    with pytest.raises(ValueError):
        EC.collate_v2(cache, [0, n])
    with pytest.raises(ValueError):
        EC.collate_v2(cache, [-1])
    with pytest.raises(ValueError):
        EC.collate_v2(cache, [3, 3])


def test_compare_batches_detects_mismatch():
    a = {"x": torch.arange(4), "n_graphs": 2}
    b = {"x": torch.arange(4), "n_graphs": 2}
    assert EC.compare_batches(a, b)["all_equal"]
    c = {"x": torch.arange(4) + 1, "n_graphs": 2}
    assert not EC.compare_batches(a, c)["all_equal"]
    d = {"x": torch.arange(4, dtype=torch.int32), "n_graphs": 2}
    assert not EC.compare_batches(a, d)["all_equal"]


# ---------------------------------------------------------------------------
# E2/E3: step equivalence, deferred log, prefetch order
# ---------------------------------------------------------------------------


def test_deferred_step_is_bitwise_equal_to_frozen_step_on_cpu():
    _require(SMOKE_TRAIN)
    task = _smoke_task(scale=1)
    rows = np.arange(8 * C.BATCH_SIZE)
    C.seed_everything(0)
    model_a, _ = M.build_model(task["spec"], task["norm"], scale=1, seed=0, device="cpu")
    opt_a, _ = M.make_optimizer(model_a)
    state0 = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
    opt0 = opt_a.state_dict()
    C.seed_everything(0)
    model_b, _ = M.build_model(task["spec"], task["norm"], scale=1, seed=0, device="cpu")
    opt_b, _ = M.make_optimizer(model_b)
    model_b.load_state_dict(state0, strict=True)
    opt_b.load_state_dict(opt0)

    clip = float(C.RECIPE["gradient_clip"])
    batches = []
    for i in range(0, rows.size, C.BATCH_SIZE):
        chunk = rows[i : i + C.BATCH_SIZE]
        batches.append(
            (
                C.PQ_DATA.collate(task["train_cache"], chunk.tolist()),
                torch.from_numpy(np.ascontiguousarray(task["train_y_z"][chunk], dtype=np.float32)),
            )
        )
    rng0_state = C.capture_rng_state()
    rng0 = C.rng_state_hashes(rng0_state)
    losses_a, gn_a = [], []
    for step, (batch, y) in enumerate(batches):
        out_a = C.PQ_TRAIN.train_step(
            model_a, opt_a, batch, y,
            global_step=step, steps_epoch=int(task["steps_epoch"]), steps_total=int(task["steps_total"]), clip=clip,
        )
        losses_a.append(out_a["loss"])
        gn_a.append(out_a["grad_norm"])
    rng_a = C.rng_state_hashes(C.capture_rng_state())
    M.restore_rng_state(rng0_state)
    losses_b, gn_b = [], []
    for step, (batch, y) in enumerate(batches):
        out_b = T.train_step_deferred(
            model_b, opt_b, batch, y,
            global_step=step, steps_epoch=int(task["steps_epoch"]), steps_total=int(task["steps_total"]), clip=clip,
        )
        losses_b.append(float(out_b["loss"]))
        gn_b.append(float(out_b["grad_norm"]))
    rng_b = C.rng_state_hashes(C.capture_rng_state())
    assert rng_a == rng_b
    assert rng_a != rng0  # dropout really did advance the generator
    assert losses_a == losses_b
    assert gn_a == gn_b
    for key in state0:
        assert torch.equal(model_a.state_dict()[key], model_b.state_dict()[key]), key



def test_deferred_log_flushes_and_counts():
    log = T.DeferredLog(check_interval=3, clip=5.0)
    for step in range(3):
        loss = torch.tensor(0.5 + step, dtype=torch.float32)
        log.add(
            {
                "loss": loss,
                "grad_norm": torch.tensor(1.0 + step),
                "bad": torch.tensor(False),
                "clipped": torch.tensor(step == 2),
                "lr": 1e-4,
                "n": 128,
            }
        )
    assert log.ready() and log.pending() == 3
    log.flush(block_seconds=0.3, first_step=0)
    assert log.losses == [0.5, 1.5, 2.5]
    assert log.grad_norms == [1.0, 2.0, 3.0]
    assert log.clip_events == 1
    assert log.n_per_step == [128, 128, 128]
    assert len(log.block_step_ms) == 3
    assert log.pending() == 0
    # a non-finite flag is reported with the exact step
    log.add(
        {
            "loss": torch.tensor(float("nan")),
            "grad_norm": torch.tensor(1.0),
            "bad": torch.tensor(True),
            "clipped": torch.tensor(False),
            "lr": 1e-4,
            "n": 128,
        }
    )
    with pytest.raises(RuntimeError, match="step 3"):
        log.flush(first_step=0)


def test_prefetcher_preserves_order_and_payload():
    _require(SMOKE_TRAIN)
    task = _smoke_task(scale=1)
    order = np.arange(6 * C.BATCH_SIZE, dtype=np.int64)
    prefetcher = PIPE.Prefetcher(
        task["train_cache"], task["train_y_z"], collator=C.PQ_DATA.collate, batch_size=C.BATCH_SIZE, depth=2
    )
    seen = []
    for rows, raw, y in prefetcher.epoch(order):
        expected = C.PQ_DATA.collate(task["train_cache"], rows.tolist())
        cmp = EC.compare_batches(raw, expected)
        assert cmp["all_equal"]
        assert np.array_equal(y, np.ascontiguousarray(task["train_y_z"][rows], dtype=np.float32))
        seen.append(rows.copy())
    prefetcher.close()
    assert len(seen) == 6
    assert np.array_equal(np.concatenate(seen), order)


def test_prefetcher_forwards_producer_errors():
    def boom(_cache, _rows):
        raise ValueError("producer exploded")

    prefetcher = PIPE.Prefetcher({}, np.zeros(4, dtype=np.float32), collator=boom, batch_size=2, depth=1)
    with pytest.raises(RuntimeError, match="producer exploded"):
        for _item in prefetcher.epoch(np.arange(4)):
            pass


# ---------------------------------------------------------------------------
# end-to-end training loop / checkpoint / resume
# ---------------------------------------------------------------------------


def test_smoke_training_loop_and_resume(tmp_path):
    _require(SMOKE_TRAIN)
    _require(SMOKE_VALID)
    task = _smoke_task(epochs=3, scale=1)
    out_dir = tmp_path / "run"
    payload = T.train(
        task,
        run="PCQM_W1_1M",
        epochs=1,
        out_dir=out_dir,
        device=torch.device("cpu"),
        implementation="optimized",
        soft_seconds=600.0,
        hard_seconds=1200.0,
        allow_partial_epochs=True,
        limit_batches=2,
    )
    assert payload["status"] == "COMPLETED"
    assert payload["parameters"] == 259_831
    assert payload["formal_run"] is False
    assert payload["fixed_epoch_reports"]["1"]["n"] == int(task["valid_rows"].size)
    assert (out_dir / "checkpoint_last.pt").exists()
    assert (out_dir / "checkpoint_epoch1.pt").exists()

    resume = torch.load(out_dir / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    resumed = T.train(
        task,
        run="PCQM_W1_1M",
        epochs=2,
        out_dir=out_dir,
        device=torch.device("cpu"),
        implementation="optimized",
        resume=resume,
        soft_seconds=600.0,
        hard_seconds=1200.0,
        allow_partial_epochs=True,
        limit_batches=2,
    )
    assert resumed["status"] == "COMPLETED"
    assert resumed["epochs_run"] == 2
    assert resumed["resume_audit"]["resumed_from_epoch"] == 1
    assert resumed["resume_audit"]["replayed_epoch_order_hashes_verified"]


def test_train_rejects_a_short_formal_horizon():
    _require(SMOKE_TRAIN)
    task = _smoke_task(scale=1)
    with pytest.raises(RuntimeError, match="formal run must use 40 epochs"):
        T.train(
            task,
            run="PCQM_W1_1M",
            epochs=2,
            out_dir=Path("/tmp/ew-should-not-exist"),
            device=torch.device("cpu"),
            implementation="optimized",
            soft_seconds=10.0,
            hard_seconds=20.0,
        )


def test_task_loader_pins_the_inherited_identities():
    _require(REAL_CACHE)
    task = DATA.load_task("PCQM_W3_1M", verify_hashes=False)
    assert task["scale"] == 3
    assert task["expected_parameters"] == 2_038_167
    assert task["n_rows"] == 1_000_000
    assert task["steps_epoch"] == 7_813
    assert task["steps_total"] == 312_520
    assert all(task["inherited_task_checks"].values())
    assert task["data_isolation"]["train_valid_overlap"] == 0
    assert task["label_norm"]["mean_eV"] == pytest.approx(5.687755523, rel=0, abs=1e-6)


def test_official_and_numpy_mae_agree_on_smoke_valid():
    _require(SMOKE_VALID)
    task = _smoke_task(scale=1)
    y = np.asarray(task["valid_y_eV"], dtype=np.float64)
    pred = y + np.linspace(-0.2, 0.2, y.size)
    report = C.PQ_TRAIN.mae_report(task["evaluator"], y, pred)
    assert report["crosscheck_delta_eV"] <= C.GATES["official_vs_numpy_mae_abs_tol"]
