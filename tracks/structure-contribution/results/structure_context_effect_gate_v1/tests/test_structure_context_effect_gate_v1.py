"""Correctness tests for structure-context-effect-gate-v1.

These tests are self-contained (no `data/`, no frozen results, no checkpoints):
they build tiny Lipophilicity raw graphs in memory and exercise the frozen-model
forward override, the bond correspondence, the reference samplers, the shuffling
control and the frozen decision rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESULT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESULT_DIR / "code"))

import structure_context_effect_gate_v1 as gate  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import bjoc_lipophilicity_adapter_v1 as lip  # noqa: E402


def _tiny_model(seed: int = 0):
    return lip.build_lipo_model(
        lip.LipoConfig(),
        seed=seed,
        organization_mode=src.NULL,
        global_mean=np.zeros(lip.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(lip.GLOBAL_DIM, dtype=np.float32),
        device="cpu",
    )


def _tiny_objects():
    smiles = ["CCO", "c1ccccc1O", "CC(=O)NC", "C1CCCCC1O"]
    return [lip.build_lipo_objects(lip.raw_from_smiles(s, i, f"c{i}")) for i, s in enumerate(smiles)]


def test_forward_with_overrides_matches_plain_forward():
    import torch

    model = _tiny_model()
    model.eval()
    batch = lip.batch_lipo_objects(_tiny_objects()[:2])
    with torch.no_grad():
        plain = model(batch)
        overridden, feats = gate.forward_with_overrides(model, batch, None)
    assert torch.equal(plain, overridden)
    assert feats["b"].shape[0] == batch["bond_feat"].shape[0]


def test_identity_override_is_zero_and_exact():
    import torch

    model = _tiny_model()
    model.eval()
    objs = _tiny_objects()
    obj = objs[0]
    enc = gate.natural_bond_encodings(model, [obj])[0]
    n_b = int(obj.bond_index.shape[0])
    batch = lip.batch_lipo_objects([obj, obj])
    with torch.no_grad():
        base, _ = gate.forward_with_overrides(model, batch, None)
        identity, feats = gate.forward_with_overrides(
            model, batch, {n_b + 0: torch.as_tensor(enc[0], dtype=torch.float32)}
        )
    assert float(abs(identity[1].item() - base[0].item())) <= 1e-6
    # the identity override leaves every bond and junction row untouched
    assert float((feats["b"][n_b : 2 * n_b] - feats["b"][:n_b]).abs().max()) <= 1e-7


def test_override_changes_only_junctions_touching_the_target_bond():
    import torch

    model = _tiny_model()
    model.eval()
    objs = _tiny_objects()
    obj = objs[1]
    donor_obj = objs[2]
    donor = gate.natural_bond_encodings(model, [donor_obj])[0][0]
    n_b = int(obj.bond_index.shape[0])
    n_j = int(obj.junc_index.shape[0])
    batch = lip.batch_lipo_objects([obj, obj])
    with torch.no_grad():
        _, feats = gate.forward_with_overrides(model, batch, {n_b + 0: torch.as_tensor(donor)})
    w_base = np.asarray(feats["w"][:n_j])
    w_int = np.asarray(feats["w"][n_j : 2 * n_j])
    jb = np.asarray(obj.junc_bond)
    expected = set(np.where((jb[:, 0] == 0) | (jb[:, 1] == 0))[0].tolist())
    changed = set(np.where(np.abs(w_int - w_base).max(axis=1) > 1e-6)[0].tolist())
    assert changed.issubset(expected)
    if expected:
        assert changed  # a real donor must move at least one incident junction
    b_base = np.asarray(feats["b"][:n_b])
    b_int = np.asarray(feats["b"][n_b : 2 * n_b])
    assert np.abs(b_int[0] - donor).max() <= 1e-6
    other = [i for i in range(n_b) if i != 0]
    if other:
        assert np.abs(b_int[other] - b_base[other]).max() <= 1e-6


def test_bond_correspondence_rdkit_vs_model():
    from rdkit import Chem

    objs = _tiny_objects()
    smiles = ["CCO", "c1ccccc1O", "CC(=O)NC", "C1CCCCC1O"]
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    corr = gate.build_bond_correspondence(objs, mols)
    assert corr["pass"]
    assert corr["stats"]["molecules_checked"] == len(smiles)
    assert corr["stats"]["bonds_checked"] == sum(int(o.bond_index.shape[0]) for o in objs)
    for obj, mapping in zip(objs, corr["per_molecule"]):
        assert len(mapping) == int(obj.bond_index.shape[0])


def test_reference_schemes_deterministic_and_target_excluded():
    import torch

    model = _tiny_model()
    model.eval()
    objs = _tiny_objects()
    enc = gate.natural_bond_encodings(model, objs)
    refs = gate.build_references(enc, objs)
    # every class observed in the tiny set has a mean and an anchor
    for c in refs["pools"]:
        assert refs["means"][c].shape == (lip.LipoConfig().bond_out,)
        assert refs["anchors"][c].shape == (lip.LipoConfig().bond_out,)
    bclass = int(objs[0].bond_type[0])
    a1 = gate.reference_for_instance(refs, enc, 0, 0, bclass, "anchor")
    a2 = gate.reference_for_instance(refs, enc, 0, 0, bclass, "anchor")
    assert np.array_equal(a1[0], a2[0])
    r1 = gate.reference_for_instance(refs, enc, 0, 0, bclass, "random")
    r2 = gate.reference_for_instance(refs, enc, 0, 0, bclass, "random")
    assert all(np.array_equal(x, y) for x, y in zip(r1, r2))
    assert 1 <= len(r1) <= gate.REF_RANDOM_DRAWS
    # donors never come from the target molecule
    pool = refs["pools"][bclass]
    uid, bidx = pool[0]
    vecs = r1
    for v in vecs:
        matches = [k for k, (u, _b) in enumerate(pool) if np.array_equal(enc[u][pool[k][1]], v)]
        assert all(pool[k][0] != uid for k in matches)


def test_shuffle_ct_preserves_group_multiset_and_size():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 4))
    s_ids = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3])
    strata = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    Xs = gate.shuffle_ct(X, s_ids, strata, 1)
    assert Xs.shape == X.shape
    for key in {(int(s), int(st)) for s, st in zip(s_ids, strata)}:
        mask = np.asarray([(int(s), int(st)) == key for s, st in zip(s_ids, strata)])
        assert np.allclose(np.sort(Xs[mask], axis=0), np.sort(X[mask], axis=0))


def test_sign_agreement_and_eta2():
    a = np.asarray([1.0, -1.0, 0.0, 2.0])
    b = np.asarray([1.0, -1.0, 0.0, -2.0])
    assert gate.sign_agreement(a, b) == 0.75
    y = np.asarray([1.0, 1.0, 3.0, 3.0])
    groups = np.asarray([0, 0, 1, 1])
    assert gate.eta2(y, groups) == pytest.approx(1.0)


def _entry(delta=0.003, ci_lo=0.001, p95=0.001, p=0.01, mae_b1=0.05):
    return {
        "B1": {"metrics": {"mae": mae_b1}},
        "delta_b2_minus_b1": delta,
        "bootstrap": {"ci_lo": ci_lo, "ci_hi": delta + 0.001},
        "shuffle": {"p95": p95, "p_shuffle": p},
    }


def _metrics(delta0=0.003, delta1=0.003, ci_lo0=0.001, ci_lo1=0.001, p950=0.001, p951=0.001):
    return {
        "seed0": {s: _entry(delta0, ci_lo0, p950) for s in gate.SCHEMES},
        "seed1": {s: _entry(delta1, ci_lo1, p951) for s in gate.SCHEMES},
    }


def _stability(min_rho=0.8, min_sign=0.8, cs_rho=0.8, cs_sign=0.8):
    return {
        "min_spearman": min_rho,
        "min_sign_agreement": min_sign,
        "cross_seed_median_spearman": cs_rho,
        "cross_seed_median_sign_agreement": cs_sign,
    }


def _diag_ok():
    return {
        f"seed{i}": {
            "w_unexpected_max": 0.0,
            "b_unexpected_max": 0.0,
            "identity_max_abs": 0.0,
            "localisation_violations": 0,
            "nonb_modified_violations": 0,
            "override_not_applied": 0,
            "identity_violations": 0,
            "molecules_checked": 10,
        }
        for i in (0, 1)
    }


def _verdict(metrics, stability=None, gate_pass=True, verification=None, diag=None, correspondence=None):
    return gate.compute_verdict(
        metrics,
        stability or _stability(),
        {"pass": gate_pass},
        verification or {"pass": True},
        diag or _diag_ok(),
        correspondence or {"pass": True},
    )["verdict"]


def test_decision_rule_signal_present():
    assert _verdict(_metrics()) == "CONTEXT_EFFECT_SIGNAL_PRESENT"


def test_decision_rule_no_justified_on_cross_seed_reproducibility():
    assert _verdict(_metrics(), _stability(cs_rho=0.4)) == "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"


def test_decision_rule_no_justified_on_reference_instability():
    assert _verdict(_metrics(), _stability(min_rho=0.3)) == "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"


def test_decision_rule_no_justified_when_never_positive():
    m = _metrics(delta0=-0.001, delta1=-0.001, ci_lo0=-0.002, ci_lo1=-0.002)
    assert _verdict(m) == "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"


def test_decision_rule_inconclusive_when_seeds_disagree():
    m = _metrics(ci_lo0=-0.001, ci_lo1=0.001)
    assert _verdict(m) == "CONTEXT_EFFECT_INCONCLUSIVE"


def test_decision_rule_inconclusive_on_support_gate_or_null():
    assert _verdict(_metrics(), gate_pass=False) == "CONTEXT_EFFECT_INCONCLUSIVE"
    m = _metrics(p950=0.01, p951=0.01, delta0=0.003, delta1=0.003)
    assert _verdict(m) == "CONTEXT_EFFECT_INCONCLUSIVE"


def test_decision_rule_fails_closed_on_implementation_problem():
    v = _verdict(_metrics(), verification={"pass": False})
    assert v == "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"
    assert _verdict(_metrics(), correspondence={"pass": False}) == "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"
    diag = _diag_ok()
    diag["seed0"]["localisation_violations"] = 1
    assert _verdict(_metrics(), diag=diag) == "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"
    diag2 = _diag_ok()
    diag2["seed1"]["w_unexpected_max"] = 1e-3
    assert _verdict(_metrics(), diag=diag2) == "NO_JUSTIFIED_CONTEXT_EFFECT_METHOD"


def test_all_cell_stability_counts_pairs():
    fake = {}
    rng = np.random.default_rng(0)
    base = rng.normal(size=50)
    for scheme in gate.SCHEMES:
        for seed in gate.MODEL_SEEDS:
            noise = rng.normal(scale=0.01, size=50) if seed == 0 else rng.normal(scale=0.6, size=50)
            fake[f"y_seed{seed}_{scheme}"] = base + noise
    st = gate._all_cell_stability(fake)
    assert st["n_pairs"] == 15
    assert set(st["all_pairs"]) and "anchor@s0__anchor@s1" in st["all_pairs"]
    assert st["min_spearman"] <= st["cross_seed_median_spearman"] + 1e-12
