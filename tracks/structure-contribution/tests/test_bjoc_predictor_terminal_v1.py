"""cscl-bjoc-predictor-terminal-evaluation-v1 tests.

Fast tests (no ZINC data, no checkpoints) cover:

* the frozen protocol file and its invariants;
* label isolation (the ``LabelGuard`` wrapper, label zeroing, and the fact
  that inference cannot reach the scoring path);
* the metric / bootstrap / decision arithmetic used by the terminal score;
* the refusal to start a second training trajectory.

Slow tests (marked ``slow``) load real frozen checkpoints and official
validation data and exercise the reuse audit, the zero-training validation
audit and a 4-molecule end-to-end prediction smoke.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import run_bjoc_predictor_terminal_v1 as term  # noqa: E402


# ---------------------------------------------------------------------------
# fast: protocol freeze
# ---------------------------------------------------------------------------


def test_protocol_yaml_is_frozen_and_complete():
    assert term.CONFIG_PATH.exists(), "frozen protocol file is missing"
    text = term.CONFIG_PATH.read_text(encoding="utf-8")
    for needle in (
        "protocol_id: cscl-bjoc-predictor-terminal-evaluation-v1",
        "count: 1",
        "MAE_N_mean: 0.5 * (MAE_N0 + MAE_N1)",
        "MAE_Y_mean: 0.5 * (MAE_Y0 + MAE_Y1)",
        "pred_N_ens: 0.5 * (pred_N0 + pred_N1)",
        "pred_Y_ens: 0.5 * (pred_Y0 + pred_Y1)",
        "molecule bootstrap, 5000 resamples",
    ):
        assert needle in text, needle


def test_four_models_are_fixed():
    assert term.MODELS == ("Y0", "Y1", "N0", "N1")
    assert term.BOOTSTRAP_N == 5000 and term.BOOTSTRAP_SEED == 0
    assert set(term.Y_SOUP) == {"Y0", "Y1"}
    assert set(term.N_SOUP) == {"N0", "N1"}


# ---------------------------------------------------------------------------
# fast: label isolation
# ---------------------------------------------------------------------------


class _FakeRow:
    def __init__(self) -> None:
        self.x = np.arange(6, dtype=np.float64).reshape(3, 2)
        self.y = torch.tensor([1.0, 2.0, 3.0])
        self.g = torch.tensor([4.0, 5.0, 6.0])
        self.c = torch.tensor([7.0])


def test_label_guard_blocks_every_label_field():
    row = _FakeRow()
    guarded = term.LabelGuard(row)
    np.testing.assert_array_equal(guarded.x, row.x)
    for name in term.LABEL_FIELDS:
        if hasattr(row, name):
            with pytest.raises(RuntimeError):
                getattr(guarded, name)


def test_zero_labels_clears_values_and_reports_no_reads():
    rows = [_FakeRow(), _FakeRow()]
    report = term._zero_labels(rows)
    assert report["label_values_read"] is False
    assert report["label_tensors_zeroed"] >= 4
    for row in rows:
        assert float(torch.abs(row.y).max()) == 0.0
        assert float(torch.abs(row.g).max()) == 0.0


def test_predict_stage_cannot_call_the_scoring_path():
    tree = ast.parse(Path(term.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "stage_predict")
    calls = {
        n.func.id
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_load_official_y" not in calls
    assert "stage_score" not in calls
    # the scoring stage is the only place allowed to read official labels
    both = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name in ("stage_score", "stage_valid_audit")
    ]
    assert len(both) == 2


# ---------------------------------------------------------------------------
# fast: metric arithmetic
# ---------------------------------------------------------------------------


def test_abs_metrics_match_hand_computation():
    pred = np.array([0.0, 1.0, 2.0, 10.0])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    out = term.abs_metrics(pred, y)
    e = np.abs(pred - y)
    assert out["mae"] == pytest.approx(e.mean())
    assert out["rmse"] == pytest.approx(np.sqrt((e ** 2).mean()))
    assert out["median_ae"] == pytest.approx(np.median(e))
    assert out["max_ae"] == pytest.approx(e.max())
    assert out["p90_ae"] == pytest.approx(np.percentile(e, 90))


def test_paired_bootstrap_is_seeded_and_centred():
    rng = np.random.default_rng(3)
    d = rng.normal(-0.01, 0.1, size=500)
    a = term.paired_bootstrap(d, n=500, seed=0)
    b = term.paired_bootstrap(d, n=500, seed=0)
    assert a == b, "the bootstrap must be reproducible for a fixed seed"
    assert a["ci95"][0] <= a["mean_diff"] <= a["ci95"][1] or a["mean_diff"] < a["ci95"][0]
    assert a["resamples"] == 500


def _synthetic_preds(n=64, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, 2.0, size=n)
    preds = {
        "Y0": y + rng.normal(0.0, 0.2, size=n),
        "Y1": y + rng.normal(0.0, 0.21, size=n),
        "N0": y + rng.normal(0.0, 0.1, size=n),
        "N1": y + rng.normal(0.0, 0.11, size=n),
    }
    return y, preds


def test_score_one_identities():
    y, preds = _synthetic_preds()
    out = term._score_one("test", y, preds)
    m = out["metrics_per_model"]
    assert out["seed_macro_mae"]["Y"] == pytest.approx(0.5 * (m["Y0"]["mae"] + m["Y1"]["mae"]))
    assert out["seed_macro_mae"]["N"] == pytest.approx(0.5 * (m["N0"]["mae"] + m["N1"]["mae"]))
    assert out["deltas"]["delta_mean"] == pytest.approx(
        0.5 * (out["deltas"]["delta0_N0_minus_Y0"] + out["deltas"]["delta1_N1_minus_Y1"])
    )
    # the paired statistic must equal the seed-MAE difference exactly
    assert out["paired_bootstrap_d_mean"]["mean_diff"] == pytest.approx(out["deltas"]["delta_mean"])
    # ensemble == fixed 0.5/0.5 average
    ens_n = 0.5 * (preds["N0"] + preds["N1"])
    assert out["ensemble_metrics"]["N"]["mae"] == pytest.approx(np.abs(ens_n - y).mean())


def test_decision_branch_selection():
    y, preds = _synthetic_preds()
    strong = term._score_one("test", y, preds)
    strong["subgroups"] = {
        "y_gt_minus3": {"delta_mean": -0.05},
        "y_le_minus3": {"delta_mean": -0.05},
        "descriptive_mae_without_single_worst_row": {"excluded_row": 0, "mae": {
            "Y0": 0.2, "Y1": 0.21, "N0": 0.1, "N1": 0.11}},
    }
    dec = term._decision(strong, strong)
    assert dec["branch"] == "G1_replicated_broad_advantage"

    worse = dict(strong)
    worse["deltas"] = {**strong["deltas"], "delta_mean": 0.01}
    assert term._decision(worse, worse)["branch"] == "G4_no_test_advantage"

    ordinary_worse = json.loads(json.dumps(strong))
    ordinary_worse["deltas"]["delta_mean"] = -0.05
    ordinary_worse["subgroups"]["y_gt_minus3"]["delta_mean"] = 0.02
    ordinary_worse["subgroups"]["descriptive_mae_without_single_worst_row"]["mae"] = {
        "Y0": 0.2, "Y1": 0.21, "N0": 0.1, "N1": 0.11}
    assert term._decision(ordinary_worse, ordinary_worse)["branch"] == "G2_tail_driven_advantage"


# ---------------------------------------------------------------------------
# fast: single-trajectory discipline
# ---------------------------------------------------------------------------


def test_train_n1_refuses_a_second_trajectory(tmp_path, monkeypatch):
    fake = tmp_path / "bjoc_null_full10k_s1_soup_state.pt"
    torch.save({"w": torch.zeros(1)}, fake)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(term, "OUT", out)
    monkeypatch.setattr(term, "N_SOUP", {**term.N_SOUP, "N1": fake})
    monkeypatch.setattr(term, "N1_SOUP_OUT", tmp_path / "seed1_last5_soup_state.pt")
    with pytest.raises(SystemExit):
        term.stage_train_n1(argparse.Namespace(device="cpu"))


def test_train_n1_refuses_when_released_copy_exists(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    existing = out / "seed1_last5_soup_state.pt"
    torch.save({"w": torch.zeros(1)}, existing)
    monkeypatch.setattr(term, "OUT", out)
    monkeypatch.setattr(term, "N_SOUP", {**term.N_SOUP, "N1": tmp_path / "missing.pt"})
    monkeypatch.setattr(term, "N1_SOUP_OUT", existing)
    with pytest.raises(SystemExit):
        term.stage_train_n1(argparse.Namespace(device="cpu"))


def test_freeze_refuses_without_formal_predictions(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(term, "OUT", out)
    monkeypatch.setattr(term, "PRED", {"valid": out / "valid_predictions.npz", "test": out / "test_predictions.npz"})
    monkeypatch.setattr(term, "FREEZE_MANIFEST", out / "terminal_freeze_manifest.json")
    with pytest.raises(SystemExit):
        term.stage_freeze(argparse.Namespace())


def test_score_refuses_without_freeze(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(term, "OUT", out)
    monkeypatch.setattr(term, "FREEZE_MANIFEST", out / "terminal_freeze_manifest.json")
    with pytest.raises(SystemExit):
        term.stage_score(argparse.Namespace())


# ---------------------------------------------------------------------------
# slow: real artifacts
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reuse_audit_on_real_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(term, "OUT", tmp_path)
    code = term.stage_reuse_audit(argparse.Namespace(device="cpu", limit=64))
    assert code == 0
    audit = json.loads((tmp_path / "reuse_audit.json").read_text())
    assert audit["hard_checks_pass"] is True
    assert audit["decision"] == "REUSE_FROZEN_MODELS"
    assert audit["Y0_Y1_state_differs"] is True
    assert audit["train_replay"]["Y0"]["ok"] and audit["train_replay"]["Y1"]["ok"]
    assert audit["BJOC_N0"]["norm"]["n_fit"] == 10000
    assert audit["BJOC_N0"]["parameter_audit"]["actual_parameters"] == 413505
    assert audit["BJOC_N0"]["forbidden_module_scan"] == []


class _TinyMol:
    def __init__(self, x, edge_index, edge_attr):
        self.x = torch.as_tensor(x)
        self.edge_index = torch.as_tensor(edge_index)
        self.edge_attr = torch.as_tensor(edge_attr)


def _tiny_dataset(n=40):
    rows = []
    for i in range(n):
        k = 3 + (i % 4)
        x = torch.tensor([[1 + (j % 3)] for j in range(k)], dtype=torch.long)
        src, dst = [], []
        for j in range(k - 1):
            src += [j, j + 1]
            dst += [j + 1, j]
        ei = torch.tensor([src, dst], dtype=torch.long)
        ea = torch.zeros(len(src), dtype=torch.long)
        rows.append(_TinyMol(x, ei, ea))
    return rows


def test_score_plumbing_with_synthetic_labels(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    n = 40
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 2.0, size=n)
    preds = {tag: y + rng.normal(0.0, 0.2, size=n) for tag in term.MODELS}
    meta = json.dumps({"y_in_file": False})
    manifest_preds = {}
    for split in ("valid", "test"):
        np.savez_compressed(
            out / f"{split}_predictions.npz", row=np.arange(n),
            **{f"{tag}_pred": preds[tag] for tag in term.MODELS}, meta=meta,
        )
        per_model = {}
        for tag in term.MODELS:
            p = out / f"{split}_{tag}_preds.npz"
            np.savez_compressed(p, model=tag, pred=preds[tag], row=np.arange(n), meta=meta)
            per_model[tag] = {"path": p.name, "sha256": term.file_sha256(p)}
        manifest_preds[split] = {
            "combined": {"path": f"{split}_predictions.npz",
                         "sha256": term.file_sha256(out / f"{split}_predictions.npz")},
            "per_model": per_model,
        }
    n1_path = term.N_SOUP["N1"] if term.N_SOUP["N1"].exists() else term.N_SOUP["N0"]
    ckpt_paths = {tag: (term.Y_SOUP[tag] if tag in term.Y_SOUP else (n1_path if tag == "N1" else term.N_SOUP[tag]))
                  for tag in term.MODELS}
    manifest = {
        "protocol_id": term.PROTOCOL_ID,
        "predictions": manifest_preds,
        "models": {
            tag: {"checkpoint": "x", "checkpoint_sha256": term.file_sha256(ckpt_paths[tag])}
            for tag in term.MODELS
        },
    }
    freeze = out / "terminal_freeze_manifest.json"
    freeze.write_text(json.dumps(manifest))
    monkeypatch.setattr(term, "OUT", out)
    monkeypatch.setattr(term, "N_SOUP", {**term.N_SOUP, "N1": ckpt_paths["N1"]})
    monkeypatch.setattr(term, "N1_SOUP_OUT", ckpt_paths["N1"])
    monkeypatch.setattr(term, "PRED", {"valid": out / "valid_predictions.npz", "test": out / "test_predictions.npz"})
    monkeypatch.setattr(term, "FREEZE_MANIFEST", freeze)
    monkeypatch.setattr(term, "_load_official_y", lambda split: y)
    monkeypatch.setattr(term, "_raw_pyg", lambda split: _tiny_dataset(n))
    monkeypatch.setattr(term, "HIST_MAE_ATOL", 1e9)
    assert term.stage_score(argparse.Namespace()) == 0
    for name in ("official_valid_metrics.json", "official_test_metrics.json", "paired_bootstrap.json",
                 "subgroup_analysis.json", "scientific_decision.json", "scoring_record.json"):
        assert (out / name).exists(), name
    scored = json.loads((out / "official_test_metrics.json").read_text())
    assert scored["n_molecules"] == n
    dec = json.loads((out / "scientific_decision.json").read_text())
    assert "Q1_seed1_training_completed" in dec["answers"]
    assert dec["branch"].startswith(("G1", "G2", "G3", "G4", "point", "partially"))
    log = json.loads((out / "scoring_record.json").read_text())
    assert log["n_label_reads"] == 2 and log["single_invocation"] is True


@pytest.mark.slow
def test_validate_valid_audit_decomposition(tmp_path, monkeypatch):
    monkeypatch.setattr(term, "OUT", tmp_path)
    code = term.stage_valid_audit(argparse.Namespace())
    assert code == 0
    payload = json.loads((tmp_path / "valid_error_decomposition.json").read_text())
    assert payload["alignment"]["ok"] is True
    gain = payload["C1_valid0172"]["gain_N0_vs_Y0"]
    assert gain["conservation_ok"] is True
    assert gain["other999_scaling_ok"] is True
    assert gain["share_of_total_gain_from_0172"] > 0.5, "the previous headline gain is row-dominated"
    assert payload["C4_method_naming_boundary"]["honest_caveat"]


@pytest.mark.slow
def test_predict_smoke_is_label_blind(tmp_path, monkeypatch):
    stand_in = tmp_path / "n1_stand_in.pt"
    shutil.copy2(term.N_SOUP["N0"], stand_in)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(term, "OUT", out)
    monkeypatch.setattr(term, "N_SOUP", {**term.N_SOUP, "N1": stand_in})
    monkeypatch.setattr(term, "N1_SOUP_OUT", stand_in)
    code = term.stage_predict(argparse.Namespace(device="cpu", limit=4))
    assert code == 0
    log = json.loads((out / "prediction_access_log.json").read_text())
    assert log["official_test_labels_read"] is False
    assert log["official_valid_labels_read"] is False
    for split in ("valid", "test"):
        entry = log["results"][split]
        assert entry["smoke"] is True
        assert entry["alignment"]["ok"] is True
        assert entry["y_independence"]["ok"] is True
        assert all(entry["finiteness"].values())
    z = np.load(out / "test_smoke_predictions.npz", allow_pickle=False)
    for tag in term.MODELS:
        assert z[f"{tag}_pred"].shape == (4,)
    meta = json.loads(str(z["meta"]))
    assert meta["y_in_file"] is False
    assert meta["official_test_labels_read"] is False
