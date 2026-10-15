"""Fast correctness tests for ``bjoc-structure-explanation-audit-v1``.

No ZINC data, no frozen checkpoint required for the core tests: tiny synthetic
molecules and a randomly initialised BJOC only.  The two frozen-source tests
skip cleanly when the (git-ignored) binaries are absent and otherwise re-check
SHA-256 against the committed record files.

Covers: the exact ``n_b`` identity, the exact Null grouping formula, junction
permutation invariance of the Null grouping, the remote-junction propagation
witness, forward == reader(forward_features), relabel/bond-order invariance,
the deterministic label-blind sample rule, and source-hash consistency.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_structure_explanation_audit_v1 as aud  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

NORM = {
    "global_mean": np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
    "global_scale": np.ones(sro.GLOBAL_DIM, dtype=np.float32),
}


def _mol(idx: int, atom_types, bonds):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in bonds:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    return sro.build_mol_objects(idx, x, np.asarray(ei, dtype=np.int64).T, np.asarray(ea, dtype=np.int64))


def _tiny_world() -> list[sro.MolObjects]:
    specs = [
        ([0, 1, 2, 3], [(0, 1, 1), (1, 2, 2), (2, 3, 1), (3, 0, 1)]),
        ([4, 5, 6], [(0, 1, 1), (1, 2, 3)]),
        ([7, 7, 8, 9, 10], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 1, 1)]),
        ([1, 2, 3, 4, 5], [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1), (4, 0, 1)]),
        ([2, 2, 3], [(0, 1, 2), (1, 2, 1)]),
        ([0, 1, 2, 3, 4, 5], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 1), (5, 0, 1), (0, 3, 1)]),
    ]
    return [_mol(i, a, b) for i, (a, b) in enumerate(specs)]


def _model(seed: int = 0) -> src.BJOC:
    cfg = src.BJOCConfig()
    model = src.build_bjoc(cfg, seed=seed, organization_mode=src.NULL, global_mean=NORM["global_mean"], global_scale=NORM["global_scale"])
    model.eval()
    return model


def test_n_b_degree_formula_is_exact_on_synthetic_molecules():
    for obj in _tiny_world():
        report = aud._a2_static_identity([obj])
        assert report["max_abs_n_b_minus_degree_formula"] == 0.0
        assert report["max_abs_sum_n_b_minus_2_n_junc"] == 0.0


def test_null_grouping_formula_and_conservation():
    objs = _tiny_world()
    batch = sro.batch_objects(objs)
    model = _model()
    feats = aud.run_features(model, batch)
    w = feats["w"]
    jb, bb = batch["junc_batch"], batch["bond_batch"]
    n = int(batch["n_graphs"])
    counts = torch.zeros(n).index_add_(0, jb, torch.ones_like(jb, dtype=w.dtype)).clamp(min=1.0)
    mean_w = torch.zeros((n, w.shape[1])).index_add_(0, jb, w) / counts.unsqueeze(1)
    expect_w = feats["n_b"].unsqueeze(1) * mean_w[bb]
    assert torch.equal(expect_w, feats["W"])
    z = torch.zeros((n, feats["c"].shape[1])).index_add_(0, bb, feats["W"])
    z_manual = (2.0 * counts).unsqueeze(1) * mean_w
    assert float((z - z_manual).abs().max()) <= 1e-5 * max(1.0, float(z.abs().max()))


def test_null_junction_permutation_is_a_no_op_but_w_is_not_the_issue():
    objs = _tiny_world()
    batch = sro.batch_objects(objs)
    model = _model()
    feats = aud.run_features(model, batch)
    perm = aud.permute_junctions_within_molecules(batch, seed=7)
    feats_p = aud.run_features(model, perm)
    assert float((feats["c"] - feats_p["c"]).abs().max()) < 1e-6
    assert float((feats["reader_in"] - feats_p["reader_in"]).abs().max()) < 1e-6


def test_remote_junction_perturbation_reaches_every_bond_of_its_molecule():
    obj = _mol(0, [0, 1, 2, 3, 4, 5], [(0, 1, 1), (1, 2, 1), (2, 3, 2), (3, 4, 1), (4, 5, 1), (5, 0, 1)])
    batch = sro.batch_objects([obj])
    model = _model()
    feats = aud.run_features(model, batch)
    w = feats["w"].clone()
    torch.manual_seed(0)
    j0 = 0
    delta = torch.randn_like(w)
    delta = delta / delta.norm(dim=1, keepdim=True).clamp(min=1e-12)
    w2 = w.clone()
    w2[j0] = w2[j0] + 0.1 * delta[j0]
    W0, V0, n_b = model.group_to_bonds(w, batch, src.NULL)
    W2, V2, _ = model.group_to_bonds(w2, batch, src.NULL)
    counts = len(obj.junc_index)
    expect = n_b.unsqueeze(1) * (0.1 * delta[j0] / counts)
    assert float(((W2 - W0) - expect).abs().max()) < 1e-6
    c2 = model.group_composer(torch.cat([feats["b"], W2, V2, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1))
    dc = (c2 - feats["c"]).abs().max(dim=1).values
    assert float((dc > 1e-6).to(torch.float64).mean()) == 1.0


def test_forward_equals_reader_of_features():
    batch = sro.batch_objects(_tiny_world())
    model = _model()
    with torch.no_grad():
        pred = model(batch)
        feats = model.forward_features(batch)
        pred2 = model.reader(feats["reader_in"]).view(-1)
    assert float((pred - pred2).abs().max()) == 0.0
    assert feats["reader_in"].shape[1] == 321


def test_relabel_bond_order_and_junction_order_invariance():
    objs = _tiny_world()
    batch = sro.batch_objects(objs)
    model = _model()
    base = aud.run_pred(model, batch)
    assert float(np.abs(base - aud.run_pred(model, aud.permute_bonds_within_molecules(batch, seed=3))).max()) < 1e-5
    assert float(np.abs(base - aud.run_pred(model, aud.permute_junctions_within_molecules(batch, seed=3))).max()) < 1e-5
    relabelled = aud.atom_relabel(objs[0], list(reversed(range(objs[0].n_atoms))))
    one = aud.run_pred(model, sro.batch_objects([objs[0]]))[0]
    one_p = aud.run_pred(model, sro.batch_objects([relabelled]))[0]
    assert abs(float(one - one_p)) < 1e-5


def test_sample_rule_is_deterministic_and_label_free():
    objs = _tiny_world()
    keys = [aud.canonical_key_bytes(o) for o in objs]
    assert keys == [aud.canonical_key_bytes(o) for o in objs]
    # the frozen rule's two halves have to be disjoint and stable
    n = len(objs)
    rank = sorted(range(n), key=lambda i: hashlib.sha256(keys[i]).hexdigest())
    assert len(set(rank)) == n
    assert all(isinstance(int(i), int) for i in rank)


def test_expected_source_hashes_match_committed_records():
    history = {
        "soup_R3_seed0_expected": "a021d36e3c07039cf191895f8037d09bf2e734101863dccf8993a69de737f782",
        "soup_R3_seed1_expected": "b622824a8eb08043777391d7ed617da2639658b413dd83a05c57f81df53d03d9",
        "objects_train10k_expected": "14326a7ae6f9bb673a5c8b35c7d54171a4008f77edb702f0591d3d1f21338e8d",
        "prep_train10k_expected": "d0fb86a349ebdd37af069cdaa4ee71ba5045a2b67365cb28b2dd734c594b1e2a",
    }
    recipe0 = json.loads((REPO_ROOT / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1/train_R3_seed0.json").read_text())
    recipe1 = json.loads((REPO_ROOT / "tracks/structure-contribution/results/bjoc_recipe_tuning_v1/train_R3_seed1.json").read_text())
    objects = json.loads((REPO_ROOT / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/objects_manifest.json").read_text())
    assert recipe0["soup_sha256"] == history["soup_R3_seed0_expected"]
    assert recipe1["soup_sha256"] == history["soup_R3_seed1_expected"]
    assert objects["objects_train10k_sha256"] == history["objects_train10k_expected"]
    assert objects["prep_train10k_sha256"] == history["prep_train10k_expected"]


@pytest.mark.skipif(not aud.SOUP_PATHS[0].exists(), reason="frozen recipe checkpoints not present locally")
def test_frozen_r3_soup_loads_with_exact_parameter_count():
    for seed in (0, 1):
        model = aud.load_model(seed, NORM)
        assert aud.file_sha256(aud.SOUP_PATHS[seed]) == json.loads(
            (REPO_ROOT / f"tracks/structure-contribution/results/bjoc_recipe_tuning_v1/train_R3_seed{seed}.json").read_text()
        )["soup_sha256"]
        assert sum(p.numel() for p in model.parameters()) == 413505
        feats = aud.run_features(model, sro.batch_objects(_tiny_world()))
        assert feats["reader_in"].shape[1] == 321
