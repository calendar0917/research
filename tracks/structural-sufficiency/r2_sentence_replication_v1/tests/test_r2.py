"""Tests for R2-SENTENCE-REPLICATION-v1.

Fast tests need no data (frozen hashes, model shape).  ``slow`` tests load the
frozen structure cache / checkpoint, mirroring the GWIC-v0 convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import r2_common as RC  # noqa: E402
import r2_data as RD  # noqa: E402
import gwic_data as GD  # noqa: E402
import gwic_model as M  # noqa: E402
import gwic_structure as S  # noqa: E402
import gwic_tensor as T  # noqa: E402


def _dummy_norm() -> dict[str, np.ndarray]:
    return {"atom_scalar_mean": np.zeros(5), "atom_scalar_scale": np.ones(5),
            "graph_mean": np.zeros(8), "graph_scale": np.ones(8)}


@pytest.fixture(scope="session")
def raw():
    return GD._load_raw()


@pytest.fixture(scope="session")
def bundle_d0():
    return RD.load_bundle("D0", "fit")


@pytest.fixture(scope="session")
def checkpoint():
    return torch.load(RC.GWIC_RESULTS / "SENTENCE_BASE_S0" / "state_best.pt",
                      map_location="cpu", weights_only=False)


def test_frozen_pins():
    details = RC.assert_frozen_sources()
    assert details["code_sha256"] == RC.GWIC_CODE_SHA256
    cfg = RC.frozen_config()
    assert cfg["arms"]["SENTENCE_BASE"] == {
        "use_words": False, "use_geometry": False,
        "role": "complete R2 sentence trunk"}


def test_parameter_count_and_no_word_modules():
    atom_dims, bond_dims = [1] * 9, [1] * 3
    # real OGB dims are needed for the embedding shapes
    atom_dims, bond_dims = GD.atom_bond_dims()
    model, info = M.build_model(atom_dims, bond_dims, 0, _dummy_norm(), use_words=False,
                                config=RC.frozen_config())
    assert int(info["parameters"]["total"]) == RC.PARAMS_TOTAL == 416177
    assert int(model.word_in_dim) == 0
    for name in ("word_joint", "word_mlp", "word_norm", "score_mlp", "q_mlp"):
        assert not hasattr(model, name)
    assert len(model.state_dict()) == RC.SENTENCE_BASE_STATE_KEYS


def test_frozen_init_hashes_match_gwic_v0():
    frozen = RC.read_json(RC.GWIC_RESULTS / "SENTENCE_BASE_S0" / "manifest.json")
    atom_dims, bond_dims = GD.atom_bond_dims()
    _, info = M.build_model(atom_dims, bond_dims, 0, _dummy_norm(), use_words=False,
                            config=RC.frozen_config())
    assert info["init_hashes"] == frozen["init_hashes"]


@pytest.mark.slow
def test_split_identity(raw):
    d0 = RD.load_split("D0", raw)
    d1 = RD.load_split("D1", raw)
    assert d0["spec"]["identity_sha256"] == RC.D0_IDENTITY
    assert d1["spec"]["identity_sha256"] == RC.D1_IDENTITY
    for loaded in (d0, d1):
        fit, ev = loaded["fit_rows"], loaded["eval_rows"]
        assert fit.size == 2856 and ev.size == 504
        assert np.intersect1d(fit, ev).size == 0
        official = np.sort(np.asarray(raw["train_idx"], np.int64))
        assert np.array_equal(np.union1d(fit, ev), official)
    assert d0["spec"]["fit_rows_sha256"] != d1["spec"]["fit_rows_sha256"]


@pytest.mark.slow
def test_d1_scaffolds_are_disjoint_and_regenerate(raw):
    regen = RD.d1_recompute_scaffolds(raw)
    assert regen["n_scaffold_map_mismatch"] == 0
    assert regen["fit_rows_equal"] and regen["heldout_rows_equal"]
    assert regen["scaffold_overlap_recomputed"] == 0
    assert regen["n_scaffolds_fit_recomputed"] == 1318
    assert regen["n_scaffolds_heldout_recomputed"] == 267


@pytest.mark.slow
def test_target_scaler_is_fit_only(bundle_d0):
    y = bundle_d0.labels[bundle_d0.fit_rows]
    assert bundle_d0.scaler["n"] == y.size == 2856
    assert bundle_d0.scaler["mean"] == pytest.approx(float(y.mean()), abs=1e-12)
    original = GD._load_train_labels

    def patched():
        payload = original()
        label = payload["label"].copy()
        mask = np.isin(payload["row_id"], bundle_d0.eval_rows)
        label[mask] += 100.0
        return {"row_id": payload["row_id"], "label": label, "sha256": payload["sha256"]}

    try:
        GD._load_train_labels = patched
        tampered = RD.load_bundle("D0", "fit")
    finally:
        GD._load_train_labels = original
    assert tampered.scaler == bundle_d0.scaler
    d1 = RD.load_bundle("D1", "fit")
    assert d1.scaler["mean"] != bundle_d0.scaler["mean"]


@pytest.mark.slow
def test_atom_scalars_fit_only_and_v0_parity(bundle_d0, raw, checkpoint):
    v0 = RD.load_bundle("D0", "v0_fit_dev")
    assert np.allclose(v0.atom_scalar_mean,
                       checkpoint["state"]["atom_scalar_mean"].numpy(), atol=1e-6)
    assert not np.allclose(bundle_d0.atom_scalar_mean, v0.atom_scalar_mean)
    assert np.allclose(bundle_d0.graph_mean,
                       checkpoint["state"]["graph_mean"].numpy(), atol=1e-6)
    fit_atoms = np.concatenate([GD.atom_scalars(GD.molecule_view(raw, int(r)))
                                for r in bundle_d0.fit_rows.tolist()], axis=0)
    assert np.allclose(bundle_d0.atom_scalar_mean, fit_atoms.mean(axis=0), atol=1e-12)


@pytest.mark.slow
def test_checkpoint_replay(bundle_d0, raw, checkpoint):
    atom_dims, bond_dims = GD.atom_bond_dims()
    model = M.load_model_from_state(atom_dims, bond_dims, checkpoint["state"],
                                    RD.norm_of(bundle_d0), use_words=False,
                                    config=RC.frozen_config(), device="cpu")
    model.eval()
    structs = S.build_all(raw, bundle_d0.eval_rows)
    pred = np.zeros(bundle_d0.eval_rows.size, np.float64)
    with torch.no_grad():
        for start in range(0, bundle_d0.eval_rows.size, 256):
            part = bundle_d0.eval_rows[start:start + 256]
            batch = T.collate(part.tolist(), structs, raw, "cpu")
            pred[start:start + part.size] = (
                model(batch, arm="SENTENCE_BASE")["yhat_std"].double().numpy())
    pred = pred * bundle_d0.scaler["std"] + bundle_d0.scaler["mean"]
    assert RC.rmse(pred, bundle_d0.labels[bundle_d0.eval_rows]) == pytest.approx(
        0.5505589717057867, abs=1e-6)


@pytest.mark.slow
def test_batch_and_relabelling_invariance(bundle_d0, raw, checkpoint):
    atom_dims, bond_dims = GD.atom_bond_dims()
    model = M.load_model_from_state(atom_dims, bond_dims, checkpoint["state"],
                                    RD.norm_of(bundle_d0), use_words=False,
                                    config=RC.frozen_config(), device="cpu")
    model.eval()
    rows = np.concatenate([bundle_d0.fit_rows[:2], bundle_d0.eval_rows[:2]])
    views = []
    for row in rows.tolist():
        view = GD.molecule_view(raw, int(row))
        view["atom_scalars"] = GD.atom_scalars(view)
        views.append(view)

    def pred_of(vs):
        batch = RD.batch_from_views(vs, "cpu")
        with torch.no_grad():
            return model(batch, arm="SENTENCE_BASE")["yhat_std"].double().numpy()

    base = pred_of(views)
    rng = np.random.default_rng(7)
    permuted = [RD.permuted_view(v, rng.permutation(int(v["n_atoms"]))) for v in views]
    reordered = [RD.permuted_view(v, np.arange(int(v["n_atoms"])),
                                  rng.permutation(int(v["n_bonds"]))) for v in views]
    assert np.max(np.abs(pred_of(permuted) - base)) < 1e-5
    assert np.max(np.abs(pred_of(reordered) - base)) < 1e-5
    assert np.max(np.abs(pred_of(list(reversed(views)))[::-1] - base)) < 1e-6
    assert np.max(np.abs(pred_of(views + reordered)[:len(views)] - base)) < 1e-6