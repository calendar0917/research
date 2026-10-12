"""Tests for cscl-bjoc-junction-regularization-v1.

The tests are split into:

* **data-free** tests (model identity, dropout semantics, parameter count,
  initialisation, invariance, save/load, source scans) that run in a fresh
  clone on CPU;
* **cache/config-dependent** tests (frozen split hashes vs the committed
  config, object-cache hash) that skip when those files are absent.

Official valid / official test are never loaded anywhere in this file.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_junction_dropout_v1 as jd  # noqa: E402
import run_bjoc_junction_regularization_v1 as drv  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "bjoc_junction_regularization_v1.yaml"
OBJECTS_CACHE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "structural_relation_composer_v1"
    / "bjoc_objects_v1.pt"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def mk_graph(n_atoms: int, edges, atom_types=None):
    x = (
        np.zeros((n_atoms, 1), dtype=np.int64)
        if atom_types is None
        else np.asarray(atom_types, dtype=np.int64).reshape(n_atoms, 1)
    )
    ei, ea = [], []
    for u, v, t in edges:
        ei.extend([(u, v), (v, u)])
        ea.extend([t, t])
    return x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64)


def build(n_atoms, edges, atom_types=None):
    return sro.build_mol_objects(-1, *mk_graph(n_atoms, edges, atom_types))


def small_batch():
    a = build(5, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 2)])
    b = build(6, [(0, 1, 1), (1, 2, 1), (2, 0, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)])
    return sro.batch_objects([a, b])


def zero_norm():
    return np.zeros(sro.GLOBAL_DIM, dtype=np.float32), np.ones(sro.GLOBAL_DIM, dtype=np.float32)


def model_for(p: float, seed: int = 0):
    gm, gs = zero_norm()
    return jd.build_model(seed=seed, junction_dropout_p=p, global_mean=gm, global_scale=gs)


# ---------------------------------------------------------------------------
# 1. model identity and capacity
# ---------------------------------------------------------------------------


def test_r0_forward_matches_frozen_backbone_bitwise():
    gm, gs = zero_norm()
    base = src.build_bjoc(src.BJOCConfig(), seed=7, organization_mode=src.NULL, global_mean=gm, global_scale=gs)
    r0 = jd.build_model(seed=7, junction_dropout_p=0.0, global_mean=gm, global_scale=gs)
    assert jd.state_dict_hash(base) == jd.state_dict_hash(r0)
    batch = small_batch()
    base.eval()
    r0.eval()
    with torch.no_grad():
        assert torch.equal(base(batch), r0(batch))


def test_parameter_counts():
    assert sum(p.numel() for p in model_for(0.0).parameters()) == 413505
    assert sum(p.numel() for p in model_for(0.1).parameters()) == 413505


def test_no_new_trainable_parameters():
    gm, gs = zero_norm()
    base = src.build_bjoc(src.BJOCConfig(), seed=0, organization_mode=src.NULL, global_mean=gm, global_scale=gs)
    r1 = model_for(0.1)
    assert sorted(n for n, _ in base.named_parameters()) == sorted(n for n, _ in r1.named_parameters())
    assert list(base.state_dict().keys()) == list(r1.state_dict().keys())


def test_init_hash_identical_across_arms():
    for seed in (0, 1, 2):
        assert jd.state_dict_hash(model_for(0.0, seed)) == jd.state_dict_hash(model_for(0.1, seed))


def test_unauthorised_probability_rejected():
    gm, gs = zero_norm()
    for bad in (0.05, 0.15, 0.2, 0.5):
        with pytest.raises(ValueError):
            jd.build_model(seed=0, junction_dropout_p=bad, global_mean=gm, global_scale=gs)
    with pytest.raises(ValueError):
        jd.JunctionDropoutConfig(junction_dropout_p=0.05)


# ---------------------------------------------------------------------------
# 2. dropout semantics
# ---------------------------------------------------------------------------


def test_eval_predictions_identical_shared_weights():
    r0, r1 = model_for(0.0), model_for(0.1)
    r1.load_state_dict(r0.state_dict())
    batch = small_batch()
    r0.eval()
    r1.eval()
    with torch.no_grad():
        assert torch.equal(r0(batch), r1(batch))


def test_r0_train_is_identity_and_rng_neutral():
    r0 = model_for(0.0)
    r0.train()
    batch = small_batch()
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        p1 = r0(batch)
        p2 = r0(batch)
    assert torch.equal(p1, p2)
    assert torch.equal(before, torch.get_rng_state())


def test_r1_train_consumes_rng_and_is_stochastic():
    r1 = model_for(0.1)
    r1.train()
    batch = small_batch()
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        p1 = r1(batch)
        p2 = r1(batch)
    assert not torch.equal(p1, p2)
    assert not torch.equal(before, torch.get_rng_state())


def test_dropout_only_on_junction_embedding(monkeypatch):
    r1 = model_for(0.1)
    r1.train()
    batch = small_batch()
    captured = {}
    orig = jd.JunctionDropoutBJOC.group_to_bonds

    def spy(self, w, batch_, mode):
        captured["w_in"] = w.detach().clone()
        return orig(self, w, batch_, mode)

    monkeypatch.setattr(jd.JunctionDropoutBJOC, "group_to_bonds", spy)
    torch.manual_seed(0)
    _, feats1 = r1(batch, return_features=True)
    a1, b1 = feats1["a"].clone(), feats1["b"].clone()
    torch.manual_seed(1)
    _, feats2 = r1(batch, return_features=True)
    assert torch.equal(a1, feats2["a"])
    assert torch.equal(b1, feats2["b"])
    w_used = captured["w_in"]
    assert torch.equal(w_used, feats2["w_used"])
    zeros = float((w_used == 0).float().mean())
    assert 0.03 <= zeros <= 0.20
    live = w_used != 0
    assert torch.allclose(w_used[live], feats2["w"][live] / (1.0 - 0.10), atol=1e-5)


def test_eval_preserves_all_junctions(monkeypatch):
    r1 = model_for(0.1)
    r1.eval()
    batch = small_batch()
    captured = {}
    orig = jd.JunctionDropoutBJOC.group_to_bonds

    def spy(self, w, batch_, mode):
        captured["w_in"] = w.detach().clone()
        return orig(self, w, batch_, mode)

    monkeypatch.setattr(jd.JunctionDropoutBJOC, "group_to_bonds", spy)
    with torch.no_grad():
        _, feats = r1(batch, return_features=True)
    assert torch.equal(captured["w_in"], feats["w"])


def test_both_moments_use_w_used(monkeypatch):
    r1 = model_for(0.1)
    r1.train()
    batch = small_batch()
    captured = {}
    orig = jd.JunctionDropoutBJOC.group_to_bonds

    def spy(self, w, batch_, mode):
        captured["w_in"] = w.detach().clone()
        return orig(self, w, batch_, mode)

    monkeypatch.setattr(jd.JunctionDropoutBJOC, "group_to_bonds", spy)
    torch.manual_seed(0)
    _, feats = r1(batch, return_features=True)
    W, V, _ = orig(r1, captured["w_in"], batch, src.NULL)
    assert torch.allclose(W, feats["W"], atol=1e-6)
    assert torch.allclose(V, feats["V"], atol=1e-6)
    W_raw, V_raw, _ = orig(r1, feats["w"], batch, src.NULL)
    assert float((W - W_raw).abs().sum()) > 0.0
    assert float((V - V_raw).abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 3. invariance, gradients, numeric health
# ---------------------------------------------------------------------------


def test_permutation_invariance_eval():
    obj = build(6, [(0, 1, 1), (1, 2, 1), (2, 0, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)])
    perm = np.random.default_rng(3).permutation(int(obj.n_atoms))
    inv = np.argsort(perm)
    obj_p = sro.MolObjects(
        mol_index=-1, smiles="permuted", n_atoms=int(obj.n_atoms),
        atom_feat=np.asarray(obj.atom_feat, dtype=np.float32)[perm],
        bond_index=inv[np.asarray(obj.bond_index, dtype=np.int64)],
        bond_type=np.asarray(obj.bond_type, dtype=np.int64),
        bond_feat=np.asarray(obj.bond_feat, dtype=np.float32),
        junc_index=inv[np.asarray(obj.junc_index, dtype=np.int64)],
        junc_bond=np.asarray(obj.junc_bond, dtype=np.int64),
        junc_feat=np.asarray(obj.junc_feat, dtype=np.float32),
        global_feat=np.asarray(obj.global_feat, dtype=np.float32),
    )
    r1 = model_for(0.1)
    r1.eval()
    with torch.no_grad():
        a = float(r1(sro.batch_objects([obj])).item())
        b = float(r1(sro.batch_objects([obj_p])).item())
    assert abs(a - b) < 1e-5


def test_batch_consistency_eval():
    objs = [build(5, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 2)]),
            build(6, [(0, 1, 1), (1, 2, 1), (2, 0, 1), (2, 3, 1), (3, 4, 1), (4, 5, 1)])]
    r1 = model_for(0.1)
    r1.eval()
    with torch.no_grad():
        both = r1(sro.batch_objects(objs))
        one = r1(sro.batch_objects([objs[0]]))
        two = r1(sro.batch_objects([objs[1]]))
    assert torch.allclose(both, torch.cat([one, two]), atol=1e-5)


def test_junction_encoder_backprop_live():
    for p in (0.0, 0.1):
        model = model_for(p)
        model.train()
        torch.manual_seed(0)
        batch = small_batch()
        model.zero_grad(set_to_none=True)
        _, feats = model(batch, return_features=True)
        feats["w"].retain_grad()
        F.l1_loss(feats["reader_in"].sum(dim=1), torch.zeros(2)).backward()
        gn = float(sum(q.grad.pow(2).sum() for q in model.junction_encoder.parameters()).sqrt())
        assert gn > 0.0
        assert feats["w"].grad is not None and float(feats["w"].grad.pow(2).sum()) > 0.0


def test_no_nan_inf_in_train_steps():
    for p in (0.0, 0.1):
        model = model_for(p)
        model.train()
        batch = small_batch()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        torch.manual_seed(0)
        for _ in range(3):
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), torch.zeros(2))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            assert np.isfinite(float(loss))


def test_label_independence():
    model = model_for(0.1)
    model.eval()
    batch = small_batch()
    batch_y = dict(batch)
    batch_y["y"] = torch.full((2,), 999.0)
    with torch.no_grad():
        assert torch.equal(model(batch), model(batch_y))


def test_save_load_replay():
    gm, gs = zero_norm()
    model = model_for(0.1)
    tmp = Path(__file__).resolve().parent / "_tmp_jdrop_state.pt"
    torch.save(model.state_dict(), tmp)
    try:
        clone = jd.build_model(seed=0, junction_dropout_p=0.1, global_mean=gm, global_scale=gs)
        clone.load_state_dict(torch.load(tmp, map_location="cpu", weights_only=False))
    finally:
        tmp.unlink()
    model.eval()
    clone.eval()
    batch = small_batch()
    with torch.no_grad():
        assert torch.equal(model(batch), clone(batch))


# ---------------------------------------------------------------------------
# 4. source / engineering-discipline scans
# ---------------------------------------------------------------------------


def test_single_dropout_location_in_model_source():
    import inspect

    body = inspect.getsource(jd.JunctionDropoutBJOC.forward_features)
    assert body.count("F.dropout(") == 1
    assert "group_to_bonds(w_used" in body
    assert "w_used = F.dropout(w" in body


def test_driver_has_no_new_dropout_variants():
    text = Path(drv.__file__).read_text(encoding="utf-8")
    assert "= F.dropout(" not in text
    assert "DropPath" not in text
    assert "drop_path" not in text


def test_official_splits_never_referenced():
    for path in (jd.__file__, drv.__file__):
        text = Path(path).read_text(encoding="utf-8")
        for forbidden in ("load_official_valid", "load_official_test", "zinc.valid", "zinc.test",
                          "valid_predictions.npz", "test_predictions.npz", "official_valid_eval"):
            assert forbidden not in text, (path, forbidden)


def test_decide_reads_no_labels():
    import inspect

    source = inspect.getsource(drv.cmd_decide)
    assert "load_y_rows" not in source
    assert "_load_y" not in source


def test_frozen_splits_reproduce_from_config():
    if not CONFIG.exists():
        pytest.skip("config absent")
    config = drv.load_config()
    smiles = drv.load_smiles()
    for key in ("A", "B"):
        split = drv.build_split(smiles, config["splits"][key]["salt"])
        for name in ("fit", "monitor", "locked"):
            assert drv.rows_sha256(split[name]) == config["splits"][key][f"{name}_sha256_rows"]


def test_split_groups_never_cross_boundaries():
    smiles = drv.load_smiles()
    for key, salt in drv.SPLIT_SALTS.items():
        split = drv.build_split(smiles, salt)
        assert split["overlap"] == {"fit_monitor": 0, "fit_locked": 0, "monitor_locked": 0, "union": 10000}
        groups: dict[str, set] = {}
        for i, s in enumerate(smiles):
            groups.setdefault(s, set()).add(i)
        for s, members in groups.items():
            if len(members) > 1:
                sides = {
                    "fit" if i in split["fit"] else "monitor" if i in split["monitor"] else "locked"
                    for i in members
                }
                assert len(sides) == 1, (key, s)


@pytest.mark.skipif(not OBJECTS_CACHE.exists(), reason="object cache absent")
def test_objects_cache_hash_matches_frozen_constant():
    digest = hashlib.sha256()
    with OBJECTS_CACHE.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    assert digest.hexdigest() == drv.OBJECTS_CACHE_SHA256


# ---------------------------------------------------------------------------
# 5. score / decide file-shape rehearsal (no real labels are read)
# ---------------------------------------------------------------------------


def test_score_and_decide_pipeline_shape(tmp_path, monkeypatch):
    """Exercise cmd_score/cmd_decide with synthetic predictions and a fake label map."""
    import json

    monkeypatch.setattr(drv, "OUT", tmp_path)
    rows = {
        "fit": list(range(0, 20)),
        "monitor": list(range(20, 30)),
        "locked": list(range(30, 40)),
    }
    monkeypatch.setattr(drv, "read_split_rows", lambda key: rows)
    ymap = {r: float((r * 37) % 11) - 3.0 for r in range(40)}
    monkeypatch.setattr(
        drv,
        "load_y_rows",
        lambda rws, subset_name="x": np.asarray([ymap[int(r)] for r in rws], dtype=np.float64),
    )

    def fake_masks(objs, rows_, y, smiles):
        return {
            "all": np.ones(len(rows_), dtype=bool),
            "y_gt_-3": y > -3.0,
            "y_le_-3": y <= -3.0,
        }

    monkeypatch.setattr(drv, "_subgroup_masks", fake_masks)
    monkeypatch.setattr(drv, "load_objects", lambda: [None] * 40)
    monkeypatch.setattr(drv, "load_smiles", lambda: ["C"] * 40)
    rng = np.random.default_rng(0)
    data = {"monitor_row": np.asarray(rows["monitor"]), "locked_row": np.asarray(rows["locked"])}
    for arm, offset in (("R0", 0.0), ("R1", -0.05)):
        for seed in (0, 1):
            tag = f"{arm}_A_s{seed}"
            data[f"pred_{tag}_monitor"] = (rng.normal(0, 0.2, 10) + offset).astype(np.float32)
            data[f"pred_{tag}_locked"] = (rng.normal(0, 0.2, 10) + offset).astype(np.float32)
            (tmp_path / f"train_{tag}.json").write_text(
                json.dumps({
                    "junction_dropout_p": 0.0 if arm == "R0" else 0.1,
                    "seed": seed,
                    "soup_fit_mae": 0.04 + 0.01 * seed,
                    "parameters": 413505,
                    "wall_seconds": 1.0,
                    "init_state_sha256": f"init{seed}",
                    "batch_order_sha256": f"order{seed}",
                }),
                encoding="utf-8",
            )
    np.savez_compressed(tmp_path / "split_a_predictions.npz", **data)
    assert drv.cmd_score(argparse_namespace(split="A")) == 0
    assert (tmp_path / "split_a_metrics.json").exists()
    assert (tmp_path / "split_a_paired_bootstrap.json").exists()
    (tmp_path / "dropout_correctness_gates.json").write_text(
        json.dumps({"all_numeric_gates_pass": True}), encoding="utf-8"
    )
    assert drv.cmd_decide(argparse_namespace(split="A")) == 0
    assert (tmp_path / "split_a_promotion_gate.json").exists()


def argparse_namespace(**kwargs):
    import argparse as _argparse

    return _argparse.Namespace(**kwargs)
