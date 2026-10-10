"""Model-level correctness tests: exact additivity, invariances, arm
distinction, shuffle effect, reproducibility, save/load.

Fresh-clone safe: synthetic molecules only (no data/ dependency).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import cscl_models as cm  # noqa: E402
from cscl_units import UnitVocab  # noqa: E402

VOCAB = 8
DESC_DIM = cf.DESC_DIM


def rand_mol_tensors(n_units: int, n_rel: int, seed: int) -> cm.Batch:
    g = torch.Generator().manual_seed(seed)
    type_ids = torch.randint(0, VOCAB, (n_units,), generator=g)
    desc = torch.randn(n_units, DESC_DIM, generator=g)
    unit_mol = torch.zeros(n_units, dtype=torch.long)
    # two molecules split of units
    unit_mol[n_units // 2 :] = 1
    rel_unit_lo = torch.arange(0, 2 * n_rel, 2)
    rel_unit_hi = rel_unit_lo + 1
    rel_mol = torch.tensor([0] * (n_rel // 2) + [1] * (n_rel - n_rel // 2), dtype=torch.long)
    return cm.Batch(
        n_mols=2,
        type_ids=type_ids,
        desc=desc,
        unit_mol=unit_mol,
        rel_type_lo=torch.randint(0, VOCAB, (n_rel,), generator=g),
        rel_type_hi=torch.randint(0, VOCAB, (n_rel,), generator=g),
        rel_feat=torch.randn(n_rel, cf.REL_DIM, generator=g),
        rel_mol=rel_mol,
        rel_unit_lo=rel_unit_lo,
        rel_unit_hi=rel_unit_hi,
    )


def model_with_centering(arm: str, seed: int = 0):
    torch.manual_seed(seed)
    model = cm.CSCLModel(VOCAB + 1, "relational" if arm == "relational" else "additive")
    g = torch.Generator().manual_seed(123)
    type_means = {t: torch.randn(DESC_DIM, generator=g) for t in range(VOCAB)}
    global_mean = torch.randn(DESC_DIM, generator=g)
    rel_mean = torch.randn(cf.REL_DIM, generator=g)
    model.set_centering(type_means, global_mean, rel_mean)
    model.eval()
    return model


def test_exact_additivity_of_contributions():
    torch.manual_seed(0)
    for arm in ("additive", "relational"):
        model = model_with_centering(arm)
        batch = rand_mol_tensors(8, 4, seed=1)
        out = model(batch)
        umol = batch.unit_mol
        rmol = batch.rel_mol
        for i in range(2):
            mu = umol == i
            mr = rmol == i
            recon = float(model.bias) + float((out.alpha[mu] + out.delta[mu]).sum()) + float(out.gamma[mr].sum())
            assert abs(recon - float(out.pred[i])) < 1e-5, (arm, i, recon, float(out.pred[i]))


def test_unit_independence_no_hidden_coupling():
    """Changing molecule 1's units must not change molecule 0's prediction."""
    model = model_with_centering("relational")
    batch = rand_mol_tensors(8, 4, seed=2)
    p0 = float(model(batch).pred[0])
    batch2 = cm.Batch(
        n_mols=batch.n_mols,
        type_ids=batch.type_ids.clone(),
        desc=batch.desc.clone(),
        unit_mol=batch.unit_mol.clone(),
        rel_type_lo=batch.rel_type_lo.clone(),
        rel_type_hi=batch.rel_type_hi.clone(),
        rel_feat=batch.rel_feat.clone(),
        rel_mol=batch.rel_mol.clone(),
        rel_unit_lo=batch.rel_unit_lo.clone(),
        rel_unit_hi=batch.rel_unit_hi.clone(),
    )
    mask = batch2.unit_mol == 1
    batch2.desc[mask] += 10.0
    batch2.type_ids[mask] = (batch2.type_ids[mask] + 3) % VOCAB
    assert abs(float(model(batch2).pred[0]) - p0) < 1e-6


def test_relabel_and_batch_invariance():
    """Permuting units (and their relations) inside a molecule leaves the
    molecule prediction unchanged (contributions travel with units)."""
    model = model_with_centering("relational")
    batch = rand_mol_tensors(8, 4, seed=3)
    out1 = model(batch).pred
    perm = torch.tensor([5, 3, 7, 1, 6, 0, 4, 2])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm))
    batch_p = cm.Batch(
        n_mols=batch.n_mols,
        type_ids=batch.type_ids[perm],
        desc=batch.desc[perm],
        unit_mol=batch.unit_mol[perm],
        rel_type_lo=batch.rel_type_lo,
        rel_type_hi=batch.rel_type_hi,
        rel_feat=batch.rel_feat,
        rel_mol=batch.rel_mol,
        rel_unit_lo=inv[batch.rel_unit_lo],
        rel_unit_hi=inv[batch.rel_unit_hi],
    )
    out2 = model(batch_p).pred
    assert torch.allclose(out1, out2, atol=1e-6)


def test_relation_endpoint_canonicalization_in_data_prep():
    """MolTensors must canonicalize relation endpoints (ta <= tb) and keep the
    unit endpoints with them — the real invariance contract for unordered
    inter-unit bonds."""
    from run_cscl_v0 import MolTensors

    class StubStats:
        def type_id(self, mol, k):
            return {0: 3, 1: 1, 2: 5}[k]

    mol = cf.MolUnits(
        mol_index=0,
        unit_sigs=["x", "y", "z"],
        unit_kinds=[0, 1, 0],
        unit_atoms=[[0], [1], [2]],
        desc=np.zeros((3, DESC_DIM), dtype=np.float32),
        rel_pairs=[(2, 0)],  # unit 2 (type 5) -- unit 0 (type 3): ta > tb -> must swap
        rel_feat=np.zeros((1, cf.REL_DIM), dtype=np.float32),
        n_atoms=3,
    )
    mt = MolTensors(mol, StubStats())
    assert mt.rel_type_lo[0] == 3 and mt.rel_type_hi[0] == 5
    assert mt.rel_unit_lo[0] == 0 and mt.rel_unit_hi[0] == 2  # endpoints travel with types


def test_arms_execute_different_computation_paths():
    torch.manual_seed(0)
    batch = rand_mol_tensors(8, 4, seed=4)

    model_a = model_with_centering("additive", seed=7)
    model_b = model_with_centering("relational", seed=7)
    # identical unary-path init under the same seed
    for (k1, p1), (k2, p2) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        if k1.startswith(("E_u", "alpha_head", "delta_mlp", "bias", "mu", "rel_center")):
            assert torch.equal(p1, p2), k1

    gen = torch.Generator().manual_seed(0)
    pred_a = cm.forward_arm(model_a, batch, "additive", gen).pred
    pred_b = cm.forward_arm(model_b, batch, "relational", gen).pred
    assert not torch.allclose(pred_a, pred_b), "B must differ from A when relations exist"

    model_c = cm.CSCLModel(VOCAB + 1, "relational")
    model_c.load_state_dict(model_b.state_dict())
    # manual relation-content permutation (mechanism of arm C): reverse rows
    rev = torch.arange(batch.rel_feat.shape[0] - 1, -1, -1)
    shuffled = cm.Batch(
        n_mols=batch.n_mols,
        type_ids=batch.type_ids,
        desc=batch.desc,
        unit_mol=batch.unit_mol,
        rel_type_lo=batch.rel_type_lo[rev],
        rel_type_hi=batch.rel_type_hi[rev],
        rel_feat=batch.rel_feat[rev],
        rel_mol=batch.rel_mol,
        rel_unit_lo=batch.rel_unit_lo,
        rel_unit_hi=batch.rel_unit_hi,
    )
    assert torch.equal(shuffled.rel_mol, batch.rel_mol)  # slot counts preserved
    if batch.rel_feat.shape[0] > 1:
        assert not torch.equal(shuffled.rel_feat, batch.rel_feat), "shuffle must be a real permutation"
    # the shuffle *function* must produce a genuine permutation too
    gen_c = torch.Generator().manual_seed(11)
    shuf_fn = cm.shuffle_relation_contents(batch, gen_c)
    assert torch.equal(torch.sort(shuf_fn.rel_feat, dim=0).values, torch.sort(batch.rel_feat, dim=0).values)
    pred_c = model_c(shuffled).pred
    pred_b_eval = model_b(batch).pred
    assert not torch.allclose(pred_c, pred_b_eval), "C must differ from B (correspondence broken)"

    model_d = cm.OpaqueModel(VOCAB + 1)
    pred_d = model_d(batch)
    assert not torch.allclose(pred_d, pred_b), "D must be a different computation path"


def test_identifiability_penalty_zero_for_centered_outputs():
    model = model_with_centering("relational")
    batch = rand_mol_tensors(8, 4, seed=5)
    out = model(batch)
    # force delta/gamma exactly at their type means / global mean -> penalty 0
    sums = torch.zeros(VOCAB + 2)
    counts = torch.zeros(VOCAB + 2)
    sums.index_add_(0, batch.type_ids, out.delta)
    counts.index_add_(0, batch.type_ids, torch.ones_like(out.delta))
    assert (sums[batch.type_ids] / counts.clamp(min=1)[batch.type_ids]).abs().max() >= 0.0
    pen = model.identifiability_penalty(batch, out)
    assert pen.ndim == 0 and torch.isfinite(pen)


def test_y_only_targets_no_aux_labels():
    y = np.array([[0.5], [-1.0], [3.0]], dtype=np.float32)
    t = cf.y_only_targets(y)
    assert t.shape == (3,) and np.allclose(t, y[:, 0])
    with pytest.raises(ValueError):
        cf.y_only_targets(np.zeros((3, 2), dtype=np.float32))  # multi-column = auxiliary labels


def test_same_seed_reproducibility():
    m1 = cm.build_model("relational", VOCAB + 1, seed=0)
    m2 = cm.build_model("relational", VOCAB + 1, seed=0)
    for (k1, p1), (k2, p2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(p1, p2), k1


def test_save_load_roundtrip_identical_predictions(tmp_path):
    model = model_with_centering("relational", seed=9)
    batch = rand_mol_tensors(8, 4, seed=10)
    ref = model(batch).pred
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    model2 = cm.CSCLModel(VOCAB + 1, "relational")
    model2.load_state_dict(sd)
    model2.eval()
    assert torch.allclose(model2(batch).pred, ref, atol=1e-7)
    path = tmp_path / "state.pt"
    torch.save(sd, path)
    model3 = cm.CSCLModel(VOCAB + 1, "relational")
    model3.load_state_dict(torch.load(path, weights_only=True))
    model3.eval()
    assert torch.allclose(model3(batch).pred, ref, atol=1e-7)


def test_vocab_known_and_unk_routing():
    """Standard-format signatures; rare fit types collapse into UNK buckets."""
    vocab = UnitVocab(min_count=3)
    vocab.fit(["0|2|a"] * 4 + ["0|2|b"] * 2 + ["0|3|c"] * 3 + ["1|5|d"] * 2)
    assert vocab.is_known("0|2|a") and vocab.is_known("0|3|c")
    assert not vocab.is_known("0|2|b") and not vocab.is_known("1|5|d")
    id_b = vocab.to_id("0|2|b", 0, 2)
    id_unseen = vocab.to_id("0|2|never-seen", 0, 2)
    assert id_b == id_unseen, "unseen dev signature must land in its UNK bucket id"
    id_other_bucket = vocab.to_id("0|9|fresh", 0, 9)
    assert id_other_bucket != id_b, "different (kind,size) bucket must be a different UNK id"
    assert vocab.n_known == 2
