"""Fast CPU tests for zinc structural-token attention v1.

These tests use synthetic label-free objects and direct model construction; they
do not load ZINC data, SSL checkpoints or official-valid labels.  The full
integration gates live in ``sta_runner.py gates`` and are run separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
for _p in (HERE, HERE.parents[2] / "structure-contribution/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import sta_data as D  # noqa: E402
import sta_models as M  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402


def _mk(n: int, edges: list[tuple[int, int]], mol_index: int = 0) -> sro.MolObjects:
    x = np.zeros((n, 1), dtype=np.int64)
    pairs = [[u, v] for u, v in edges] + [[v, u] for u, v in edges]
    ei = np.asarray(pairs, dtype=np.int64).T.reshape(2, len(pairs))
    ea = np.zeros((len(pairs),), dtype=np.int64)
    return sro.build_mol_objects(mol_index, x, ei, ea, "synthetic")


def _objects() -> list[sro.MolObjects]:
    return [
        _mk(4, [(0, 1), (1, 2), (2, 3)], 0),
        _mk(4, [(0, 1), (0, 2), (0, 3)], 1),
        _mk(1, [], 2),
        _mk(2, [(0, 1)], 3),
    ]


def _model(arm: str) -> M.STAObjectModel:
    cfg = M.src.BJOCConfig()
    model = M.STAObjectModel(
        cfg,
        np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        np.ones(sro.GLOBAL_DIM, dtype=np.float32),
        arm,
    )
    M.init_new_modules(model, 0)
    return model


@pytest.fixture(scope="module")
def toy():
    objs = _objects()
    cache = D.build_relation_cache(objs)
    batch = D.batch_objects_sta(objs, list(range(len(objs))), cache)
    return objs, cache, batch


def test_parameter_counts_and_reader():
    expected = {"S": 483137, "T": 483137, "R": 483289}
    for arm, n in expected.items():
        model = _model(arm)
        assert sum(p.numel() for p in model.parameters()) == n
        audit = M.parameter_audit(model)
        assert audit["parameter_exact"]
        assert audit["groups"]["reader"] == 68257
        expected_bias = M.RELATION_PARAMETERS if arm == "R" else 0
        assert audit["groups"]["blocks"] == 2 * M.BLOCK_PARAMETERS + expected_bias
        assert audit["groups"]["struct_bias"] == expected_bias
        assert audit["groups"]["type_embedding"] == 288
        assert not audit["old_reader_present"]
    assert M.RELATION_PARAMETERS == 152


def test_z_p3_bypass_matches_original(toy):
    _, _, batch = toy
    base = _model("S")
    # Build an original BJOC with the same shared tensors as the S model.
    cfg = M.src.BJOCConfig()
    orig = M.src.BJOC(
        cfg,
        np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        np.ones(sro.GLOBAL_DIM, dtype=np.float32),
        M.src.NULL,
    )
    with torch.no_grad():
        for key, value in base.state_dict().items():
            if key.startswith(("atom_encoder.", "bond_encoder.", "junction_encoder.", "group_composer.")):
                assert key in orig.state_dict()
                orig.state_dict()[key].copy_(value)
    with torch.no_grad():
        a = orig.forward_features(batch)["reader_in"]
        b = base.forward_features(batch)["reader_in_p3"]
    assert a.shape == b.shape == (len(_objects()), 321)
    assert torch.equal(a, b)


def test_t_r_zero_bias_identity(toy):
    _, _, batch = toy
    t = _model("T")
    r = _model("R")
    with torch.no_grad():
        ts = t.state_dict()
        rs = r.state_dict()
        for k in rs:
            if k in ts:
                rs[k].copy_(ts[k])
        r.load_state_dict(rs, strict=True)
        for block in r.blocks:
            block.mix.struct_bias.type_bias.zero_()
            block.mix.struct_bias.dist_bias.zero_()
            block.mix.struct_bias.overlap_bias.zero_()
    with torch.no_grad():
        assert torch.equal(t(batch), r(batch))


def test_relation_bins_hand_checked_path(toy):
    objs, _, _ = toy
    rec = D.relation_tables(objs[0])
    # A0..A3, B0..B2, J0..J1
    assert (int(rec["dist_bin"][0, 3]), int(rec["overlap_bin"][0, 3])) == (3, 0)
    assert (int(rec["dist_bin"][4, 6]), int(rec["overlap_bin"][4, 6])) == (1, 0)
    assert (int(rec["dist_bin"][4, 5]), int(rec["overlap_bin"][4, 5])) == (0, 1)
    assert (int(rec["dist_bin"][7, 8]), int(rec["overlap_bin"][7, 8])) == (0, 2)
    assert (int(rec["dist_bin"][3, 7]), int(rec["overlap_bin"][3, 7])) == (1, 0)
    assert (int(rec["dist_bin"][0, 8]), int(rec["overlap_bin"][0, 8])) == (1, 0)


def test_disconnected_distance_bin():
    obj = sro.MolObjects(
        mol_index=0,
        smiles="CC.CC",
        n_atoms=4,
        atom_feat=np.zeros((4, sro.ATOM_FEAT_DIM), dtype=np.float32),
        bond_index=np.asarray([[0, 1], [2, 3]], dtype=np.int64),
        bond_type=np.zeros(2, dtype=np.int64),
        bond_feat=np.zeros((2, sro.BOND_FEAT_DIM), dtype=np.float32),
        junc_index=np.zeros((0, 3), dtype=np.int64),
        junc_bond=np.zeros((0, 2), dtype=np.int64),
        junc_feat=np.zeros((0, sro.JUNC_FEAT_DIM), dtype=np.float32),
        global_feat=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    )
    rec = D.relation_tables(obj)
    assert int(rec["dist_bin"][0, 5]) == D.DIST_DISCONNECTED


def test_empty_boundaries(toy):
    objs, cache, _ = toy
    model = _model("R")
    for idx in (2, 3):
        batch = D.batch_objects_sta([objs[idx]], [0], D.build_relation_cache([objs[idx]]))
        with torch.no_grad():
            pred, feats = model(batch, return_features=True)
        assert torch.isfinite(pred).all()
        assert torch.isfinite(feats["reader_in"]).all()
        # no junction in either synthetic object
        assert float(feats["type_sums"][0, 2].abs().max()) == 0.0


def test_token_permutation_with_relation_sync(toy):
    _, _, batch = toy
    model = _model("R")
    g, t = batch["token_valid"].shape
    torch.manual_seed(3)
    h0 = torch.randn(g, t, M.DIM) * batch["token_valid"].unsqueeze(-1)
    with torch.no_grad():
        _, sums0, _ = model.encode_tokens(
            h0,
            batch["token_valid"],
            batch["token_type_padded"],
            batch["rel_dist"].long(),
            batch["rel_overlap"].long(),
        )
    perm = torch.stack([torch.randperm(t, generator=torch.Generator().manual_seed(11 + i)) for i in range(g)])
    h0p = torch.stack([h0[i, perm[i]] for i in range(g)])
    validp = torch.stack([batch["token_valid"][i, perm[i]] for i in range(g)])
    typep = torch.stack([batch["token_type_padded"][i, perm[i]] for i in range(g)])
    distp = torch.stack([batch["rel_dist"][i][perm[i]][:, perm[i]] for i in range(g)]).long()
    overp = torch.stack([batch["rel_overlap"][i][perm[i]][:, perm[i]] for i in range(g)]).long()
    with torch.no_grad():
        _, sums1, _ = model.encode_tokens(h0p, validp, typep, distp, overp)
    assert float((sums0 - sums1).abs().max()) < 1e-6


def test_batch_composition_and_cross_molecule(toy):
    objs, cache, _ = toy
    model = _model("R")
    model.eval()
    with torch.no_grad():
        single = torch.stack(
            [model(D.batch_objects_sta([o], [0], D.build_relation_cache([o]))) for o in objs]
        ).view(-1)
        mixed = model(D.batch_objects_sta(objs, list(range(len(objs))), cache))
    assert float((single - mixed).abs().max()) < 2e-5
    # perturb molecule 1 only; molecule 0 must not move
    objs2 = [objs[0], objs[1]]
    objs2[1].atom_feat = objs2[1].atom_feat + 0.5
    with torch.no_grad():
        pert = model(D.batch_objects_sta(objs2, [0, 1], D.build_relation_cache(objs2)))
    assert abs(float(pert[0]) - float(mixed[0])) < 2e-5


def test_two_update_gradients(toy):
    _, _, batch = toy
    model = _model("R")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    y = torch.linspace(-1.0, 1.0, len(_objects()))
    probes = []
    for _ in range(2):
        loss = torch.nn.functional.l1_loss(model(batch), y)
        opt.zero_grad()
        loss.backward()
        grads = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads[name] = float(p.grad.abs().sum())
        assert all(np.isfinite(v) for v in grads.values())
        assert grads.get("type_embedding", 0.0) > 0.0
        assert any(v > 0.0 for k, v in grads.items() if k.startswith("reader."))
        probes.append(grads)
        opt.step()
    r_grads = [v for k, v in probes[1].items() if ".mix.struct_bias." in k]
    assert any(v > 0.0 for v in r_grads)
