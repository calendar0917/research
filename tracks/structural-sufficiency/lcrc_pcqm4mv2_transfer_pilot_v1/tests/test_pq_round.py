"""Unit tests for the PCQM4Mv2 transfer pilot (CPU only, no formal data needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import pq_common as C  # noqa: E402
import pq_data as D  # noqa: E402

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# frozen numbers
# ---------------------------------------------------------------------------


def test_recipe_is_the_preregistered_one():
    assert C.BATCH_SIZE == 128
    assert C.TOTAL_EPOCHS == 40
    assert C.MODEL_SEED == 0 and C.ORDER_SEED == 0 and C.SAMPLE_SEED == 42 and C.MONITOR_SEED == 43
    assert C.RECIPE["learning_rate"] == 3e-4
    assert C.RECIPE["warmup_start_lr"] == 3e-5
    assert C.RECIPE["end_lr"] == 1e-5
    assert C.RECIPE["weight_decay"] == 1e-3
    assert C.RECIPE["dropout"] == 0.10
    assert C.RECIPE["gradient_clip"] == 5.0
    assert tuple(C.RECIPE["adam_betas"]) == (0.9, 0.999)
    assert C.RECIPE["adam_eps"] == 1e-8
    assert C.RECIPE["early_stopping"] is False
    assert set(C.RUNS_TABLE) == {"PCQM_W2_300K", "PCQM_W2_1M"}
    assert C.RUNS_TABLE["PCQM_W2_300K"]["train_size"] == 300_000
    assert C.RUNS_TABLE["PCQM_W2_1M"]["train_size"] == 1_000_000
    assert all(v["epochs"] == 40 and v["seed"] == 0 for v in C.RUNS_TABLE.values())


def test_widths_are_w2_everywhere_except_output():
    cfg = C.MODEL_CFG
    reference = C.BS.config_for_scale(2)
    assert cfg["n_out"] == 1
    for key, value in reference.items():
        if key == "n_out":
            assert value == 128
            continue
        assert cfg[key] == value, key
        if key in C.REFERENCE_WIDTHS:
            assert cfg[key] == 2 * C.REFERENCE_WIDTHS[key], key
    assert cfg["n_heads"] == 4


def test_expected_parameter_count_is_stable():
    import feature_spec as FS

    spec = FS.lipo_v1_spec()
    expected = C.expected_parameter_count(
        C.MODEL_CFG,
        atom_feat_dim=spec.atom_feat_dim,
        bond_feat_dim=spec.bond_feat_dim,
        n_scalars=spec.n_scalars,
        global_dim=spec.global_dim,
        n_nbr_codes=spec.n_nbr_codes,
        n_path_codes=spec.n_path_codes,
    )
    # recomputed independently from the width table
    assert expected["total_parameters"] == expected["base_parameters"] + expected["readout_parameters"]
    assert expected["local_input_dim"] == 3 * 192 + 4 * 64 + 8 + 9
    assert expected["rho_input_dim"] == spec.global_dim + 2 * 128 + 1
    assert expected["total_parameters"] == 939_079
    assert expected["total_parameters"] < expected["historical_pcba_w2_parameters"]
    assert expected["historical_pcba_w2_parameters"] - expected["total_parameters"] == 55_245


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------


def test_lr_schedule_shape_and_endpoints():
    for n in (300_000, 1_000_000):
        s = C.steps_per_epoch(n)
        t = C.total_steps(n)
        assert s == int(np.ceil(n / 128))
        assert t == 40 * s
        assert C.scheduled_lr(0, steps_epoch=s, steps_total=t) == pytest.approx(3e-5 + (3e-4 - 3e-5) / s)
        assert C.scheduled_lr(s - 1, steps_epoch=s, steps_total=t) == pytest.approx(3e-4)
        assert C.scheduled_lr(s, steps_epoch=s, steps_total=t) == pytest.approx(3e-4)
        assert C.scheduled_lr(t, steps_epoch=s, steps_total=t) == pytest.approx(1e-5)
        warm = [C.scheduled_lr(i, steps_epoch=s, steps_total=t) for i in range(0, s)]
        assert all(b > a for a, b in zip(warm, warm[1:]))
        tail = [C.scheduled_lr(i, steps_epoch=s, steps_total=t) for i in range(s, t + 1, max(1, t // 200))]
        assert all(b <= a + 1e-12 for a, b in zip(tail, tail[1:]))
        assert min(tail) >= 1e-5 - 1e-12


def test_steps_per_epoch_ceil_not_floor():
    assert C.steps_per_epoch(300_000) == 2344
    assert C.steps_per_epoch(1_000_000) == 7813
    assert C.total_steps(300_000) == 93_760
    assert C.total_steps(1_000_000) == 312_520


# ---------------------------------------------------------------------------
# subsets
# ---------------------------------------------------------------------------


def test_subset_plan_prefix_containment():
    n_total = 50_000
    info = {
        "split": {
            "train": np.arange(n_total, dtype=np.int64),
            "valid": np.arange(n_total, n_total + 100, dtype=np.int64),
            "test-dev": np.arange(n_total + 100, n_total + 200, dtype=np.int64),
            "test-challenge": np.arange(n_total + 200, n_total + 300, dtype=np.int64),
        }
    }
    plan = D.subset_plan(info, sizes={"PCQM_W2_300K": 300, "PCQM_W2_1M": 1000})
    rows = plan.pop("_permuted_rows")
    assert rows.size == n_total
    assert np.unique(rows).size == rows.size
    assert plan["containment"]["small_is_prefix_of_large"] is True
    assert plan["containment"]["no_validation_row_in_large"] is True
    assert plan["containment"]["no_hidden_row_in_large"] is True
    assert plan["status"] == "OK"
    # deterministic across calls
    plan2 = D.subset_plan(info, sizes={"PCQM_W2_300K": 300, "PCQM_W2_1M": 1000})
    assert plan2["permutation_rows_sha256_full"] == plan["permutation_rows_sha256_full"]


def test_monitor_rows_are_a_fixed_subset():
    a = D.monitor_rows()
    b = D.monitor_rows()
    assert a["n"] == C.N_MONITOR == 20_000
    assert a["positions_sha256"] == b["positions_sha256"]
    assert int(a["_positions"].max()) < 300_000
    assert np.unique(a["_positions"]).size == a["n"]
    assert np.all(np.diff(a["_positions"]) > 0)


def test_label_artifact_masks_hidden_rows(tmp_path):
    info = {
        "split": {
            "train": np.array([0, 1, 4]),
            "valid": np.array([2]),
            "test-dev": np.array([3]),
            "test-challenge": np.array([5]),
        },
        "labels": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    }
    event = D.build_label_artifact(info, tmp_path / "labels.npy")
    saved = np.load(tmp_path / "labels.npy")
    assert np.isfinite(saved[[0, 1, 2, 4]]).all()
    assert np.isnan(saved[[3, 5]]).all()
    assert event["hidden_row_values_read"] is False
    assert event["hidden_row_values_used_in_any_metric"] is False


# ---------------------------------------------------------------------------
# batch order
# ---------------------------------------------------------------------------


def test_epoch_orders_are_deterministic_and_per_epoch():
    a = D.epoch_orders(500, seed=0, epochs=3)
    b = D.epoch_orders(500, seed=0, epochs=3)
    assert a["epoch_permutation_sha256"] == b["epoch_permutation_sha256"]
    assert len(set(a["epoch_permutation_sha256"])) == 3
    batches = D.batches_for_epoch(500, 2, seed=0, batch_size=128)
    assert sum(len(x) for x in batches) == 500
    assert min(int(x.min()) for x in batches) >= 0
    assert max(int(x.max()) for x in batches) < 500
    flat = np.concatenate(batches)
    assert np.unique(flat).size == 500


# ---------------------------------------------------------------------------
# label normalisation
# ---------------------------------------------------------------------------


def test_label_norm_roundtrip():
    rng = np.random.default_rng(0)
    gap = rng.normal(5.7, 1.2, size=1000)
    norm = D.label_norm(gap)
    z = D.to_z(gap, norm)
    assert abs(z.mean()) < 1e-12
    assert abs(z.std() - 1.0) < 1e-12
    back = D.from_z(z, norm)
    assert np.allclose(back, gap, atol=1e-12)


# ---------------------------------------------------------------------------
# merge equivalence with the pinned PCBA implementation
# ---------------------------------------------------------------------------


def _tiny_shards(tmp_path, smiles):
    """Build shards through the frozen adapter chain for a handful of SMILES."""
    import feature_spec as FS
    import lcrc_data as PREP_DATA
    import mb2_rdkit as R

    spec = FS.lipo_v1_spec()
    out = []
    for i, group in enumerate([smiles[:2], smiles[2:]]):
        mols = []
        gids = []
        for row, smi in zip(range(i * 2, i * 2 + len(group)), group):
            raw, _mode = R.smiles_to_raw_row_relaxed(smi, row)
            graph = R.raw_row_to_graph(raw, row, spec)
            mols.append(PREP_DATA.build_molecule(graph, spec, validate=True))
            gids.append(row)
        cache = PREP_DATA.concat_molecules(mols, spec, split="test")
        cache["gids"] = np.asarray(gids, dtype=np.int64)
        path = tmp_path / f"shard_{i:05d}.pt"
        torch.save(cache, path)
        out.append(path)
    return spec, out


def test_merge_matches_pinned_pcba_merge(tmp_path):
    pytest.importorskip("rdkit")
    import pc_data as PCD

    smiles = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "N#Cc1ccccc1", "C1CC1", "O=C=O"]
    spec, shards = _tiny_shards(tmp_path, smiles)
    mine = D.merge_shards(shards, spec, out_dir=tmp_path / "mine", split="test", tag="t")
    theirs = PCD.merge_shards(shards, spec, out_dir=tmp_path / "theirs", split="test")
    import json

    a = D.load_cache(tmp_path / "mine")
    b = PCD.load_cache(tmp_path / "theirs")
    for key in ("atom_ptr", "bond_ptr", "nbr_ptr", "path_ptr", "atom_feat", "atom_scalar", "bond_feat",
                "bond_type", "bond_index", "nbr_code", "path_cat", "path_end63", "path_end",
                "path_closure", "global", "gids"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    assert mine["cache_bytes"] > 0
    meta = json.loads((tmp_path / "mine" / "meta.json").read_text())
    assert meta["dataset"] == C.DATASET
    assert meta["labels_present"] is False
    assert meta["feature_spec_hash"] == spec.hash()
    assert mine["cache_sha256"] != theirs["cache_sha256"]  # meta.json differs by dataset label


# ---------------------------------------------------------------------------
# collate + model forward / backward / permutation invariance
# ---------------------------------------------------------------------------


def _tiny_cache(tmp_path, smiles):
    import feature_spec as FS
    import lcrc_data as PREP_DATA
    import mb2_rdkit as R

    spec = FS.lipo_v1_spec()
    mols = []
    for row, smi in enumerate(smiles):
        raw, _mode = R.smiles_to_raw_row_relaxed(smi, row)
        graph = R.raw_row_to_graph(raw, row, spec)
        mols.append(PREP_DATA.build_molecule(graph, spec, validate=True))
    cache = PREP_DATA.concat_molecules(mols, spec, split="tiny")
    cache["gids"] = np.arange(len(smiles), dtype=np.int64)
    cache["n_molecules"] = len(smiles)
    cache["meta"] = {"labels_present": False, "feature_spec_hash": spec.hash(), "split": "tiny"}
    return spec, cache


def test_model_forward_backward_and_permutation_invariance(tmp_path):
    pytest.importorskip("rdkit")
    import pq_model as M
    import lcrc_data as PREP_DATA

    smiles = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "N#Cc1ccccc1", "C1CCC1"]
    spec, cache = _tiny_cache(tmp_path, smiles)
    rows = list(range(len(smiles)))
    atoms = np.asarray(cache["atom_scalar"], dtype=np.float64)
    glob = np.asarray(cache["global"], dtype=np.float64)
    norm = {
        "global_mean": glob.mean(0).astype(np.float32),
        "global_scale": np.where(glob.std(0) > 1e-6, glob.std(0), 1.0).astype(np.float32),
        "scalar_mean": atoms.mean(0).astype(np.float32),
        "scalar_scale": np.where(atoms.std(0) > 1e-6, atoms.std(0), 1.0).astype(np.float32),
    }
    C.seed_everything(0)
    model, audit = M.build_model(spec, norm, seed=0, device="cpu")
    assert audit["parameter_check"].startswith("exact")
    assert audit["readout_output_shape"] == [1, 176]
    assert audit["total_parameters"] == 939_079
    batch = PREP_DATA.collate(cache, rows)
    model.eval()
    with torch.inference_mode():
        z = M.predict_z(model, batch)
    assert tuple(z.shape) == (len(smiles),)
    assert torch.isfinite(z).all()
    perm_batch, _p = M.permute_atoms(batch, seed=1)
    with torch.inference_mode():
        z_perm = M.predict_z(model, perm_batch)
    assert torch.allclose(z, z_perm, atol=1e-4, rtol=0)
    # one encode + one local call per forward
    calls = M.count_module_calls(model, batch)
    assert all(v == 1 for v in calls.values()), calls
    # backward + optimizer step
    opt, _ = M.make_optimizer(model)
    model.train()
    y = torch.zeros(len(smiles))
    out = __import__("pq_train").train_step(
        model, opt, batch, y, global_step=0, steps_epoch=10, steps_total=100, clip=5.0
    )
    assert np.isfinite(out["loss"]) and out["grad_norm"] >= 0
    assert out["lr"] == pytest.approx(C.scheduled_lr(0, steps_epoch=10, steps_total=100))


def test_historical_replay_is_994324(tmp_path):
    pytest.importorskip("rdkit")
    import pq_model as M

    spec, cache = _tiny_cache(tmp_path, ["CCO", "c1ccccc1"])
    glob = np.asarray(cache["global"], dtype=np.float64)
    atoms = np.asarray(cache["atom_scalar"], dtype=np.float64)
    norm = {
        "global_mean": glob.mean(0).astype(np.float32),
        "global_scale": np.where(glob.std(0) > 1e-6, glob.std(0), 1.0).astype(np.float32),
        "scalar_mean": atoms.mean(0).astype(np.float32),
        "scalar_scale": np.where(atoms.std(0) > 1e-6, atoms.std(0), 1.0).astype(np.float32),
    }
    replay = M.historical_replay(spec, norm, seed=0)
    assert replay["matches_historical"] is True
    assert replay["parameters"] == 994_324


def test_init_is_identical_for_two_builds(tmp_path):
    pytest.importorskip("rdkit")
    import pq_model as M

    spec, cache = _tiny_cache(tmp_path, ["CCO", "c1ccccc1"])
    glob = np.asarray(cache["global"], dtype=np.float64)
    atoms = np.asarray(cache["atom_scalar"], dtype=np.float64)
    norm = {
        "global_mean": glob.mean(0).astype(np.float32),
        "global_scale": np.where(glob.std(0) > 1e-6, glob.std(0), 1.0).astype(np.float32),
        "scalar_mean": atoms.mean(0).astype(np.float32),
        "scalar_scale": np.where(atoms.std(0) > 1e-6, atoms.std(0), 1.0).astype(np.float32),
    }
    C.seed_everything(0)
    _m1, a1 = M.build_model(spec, norm, seed=0, device="cpu")
    C.seed_everything(0)
    _m2, a2 = M.build_model(spec, norm, seed=0, device="cpu")
    assert a1["state_fingerprint"] == a2["state_fingerprint"]
    assert a1["trunk_init_fingerprint"] == a2["trunk_init_fingerprint"]


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------


def test_official_and_numpy_mae_agree():
    import pq_train as T

    rng = np.random.default_rng(7)
    yt = rng.normal(5.5, 1.1, size=500)
    yp = yt + rng.normal(0, 0.2, size=500)
    report = T.mae_report(D._evaluator(), yt, yp)
    assert report["mae_eV"] == pytest.approx(report["mae_eV_numpy"], abs=1e-12)
    assert report["crosscheck_delta_eV"] <= 1e-12
    assert report["n"] == 500


def test_mae_report_rejects_non_finite():
    import pq_train as T

    with pytest.raises(ValueError):
        T.mae_report(D._evaluator(), np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_evaluator_imports_without_rdkit():
    """GPU hosts have no rdkit; the official evaluator must still import."""
    import subprocess
    import textwrap

    code = textwrap.dedent(
        f"""
        import sys
        class _Block:
            def find_spec(self, name, path=None, target=None):
                if name == "rdkit" or name.startswith("rdkit."):
                    raise ImportError("rdkit blocked")
                return None
        sys.meta_path.insert(0, _Block())
        sys.path.insert(0, {str(CODE)!r})
        import ogb.utils
        assert not hasattr(ogb.utils, "smiles2graph"), "rdkit block did not take effect"
        import pq_data as D
        ev = D._evaluator()
        assert type(ev).__module__ == "ogb.lsc.pcqm4mv2", type(ev).__module__
        import numpy as np
        out = ev.eval({{"y_pred": np.array([1.0, 2.0]), "y_true": np.array([1.5, 2.0])}})["mae"]
        assert abs(out - 0.25) < 1e-12, out
        print("OK")
        """
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr[-3000:]
    assert "OK" in res.stdout


def test_analysis_aggregates_runs_and_reports_trends(monkeypatch, tmp_path):
    """The report table must be derived from the recorded curves, not retyped."""
    import csv

    import pq_analysis as A

    monkeypatch.setattr(C, "RUNS", tmp_path)
    monkeypatch.setattr(C, "ANALYSIS", tmp_path / "analysis")
    (tmp_path / "analysis").mkdir()
    C.write_json(
        tmp_path / "analysis" / "pcqm_constant_baseline.json",
        {"valid_mae_eV": 0.98, "median_eV": 5.58},
    )
    spec = {
        "PCQM_W2_300K": {"rows": 300_000, "maes": [0.40, 0.30, 0.25, 0.24]},
        "PCQM_W2_1M": {"rows": 1_000_000, "maes": [0.35, 0.26, 0.22, 0.21]},
    }
    for run, info in spec.items():
        run_dir = tmp_path / run
        run_dir.mkdir()
        rows = []
        for i, mae in enumerate(info["maes"], start=1):
            rows.append(
                {
                    "epoch": i,
                    "evaluated": "True",
                    "global_step": i * 10,
                    "steps_per_epoch": 10,
                    "learning_rate": 3e-4,
                    "train_l1_z": 0.5 / i,
                    "valid_mae_eV": mae,
                    "valid_mae_eV_numpy": mae,
                    "valid_mae_crosscheck_delta_eV": 0.0,
                    "monitor_mae_eV": mae + 0.01,
                    "epoch_train_seconds": 60.0,
                    "epoch_valid_seconds": 5.0,
                    "epoch_monitor_seconds": 1.0,
                    "epoch_seconds": 66.0,
                    "cumulative_seconds": 66.0 * i,
                    "step_ms_p50": 50.0,
                    "grad_norm_max": 3.0,
                    "gradient_clip_events": 0,
                }
            )
        with open(run_dir / "train_curve.csv", "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        C.write_json(
            run_dir / "run.json",
            {
                "status": "COMPLETED",
                "run": run,
                "train_size": info["rows"],
                "epochs": 4,
                "epochs_run": 4,
                "steps_per_epoch": 10,
                "steps_total": 40,
                "global_steps_run": 40,
                "parameters": 939_079,
                "base_parameters": 815_330,
                "readout_parameters": 123_749,
                "best_epoch": 4,
                "best_valid_mae_eV": min(info["maes"]),
                "gpu_seconds": 3600.0,
                "gpu_hours": 1.0,
                "epoch_seconds_mean": 66.0,
                "peak_rss_mb": 1200.0,
                "git_commit": "test",
                "device": "cuda",
                "test_evaluated": False,
                "data_isolation": {"valid_rows": 100, "monitor_rows": 20},
            },
        )
    summary = A.build_summary()
    a = summary["runs"]["PCQM_W2_300K"]
    b = summary["runs"]["PCQM_W2_1M"]
    assert a["train_rows"] == 300_000 and b["train_rows"] == 1_000_000
    assert a["best_valid_mae_eV"] == pytest.approx(0.24)
    assert b["valid_mae_at_last_epoch_eV"] == pytest.approx(0.21)
    assert summary["improvement_300k_to_1m"]["best_valid_mae_delta_eV"] == pytest.approx(0.03)
    assert a["trend_last10_valid"]["slope_eV_per_epoch"] < 0
    table = A.table_markdown(summary)
    assert "Constant median" in table and "LCRC-W2-300K" in table and "LCRC-W2-1M" in table
    assert "0.2400" in table and "0.2100" in table
    assert len(A.write_curves(summary["runs"]).read_text(encoding="utf-8").splitlines()) == 9
