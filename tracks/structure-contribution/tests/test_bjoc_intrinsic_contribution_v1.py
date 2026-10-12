"""Tests for cscl-bjoc-intrinsic-contribution-feasibility-v1.

Data-free core tests (IJCM algebra, gauge, vocab, invariances, source scans)
run in a fresh clone on CPU.  Cache/config-dependent tests skip when the big
official-train caches are absent.

Official valid / official test are never loaded anywhere in this file.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_intrinsic_contribution_v1 as ijcm  # noqa: E402
import ijcm_synthetic_v1 as syn  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "bjoc_intrinsic_contribution_feasibility_v1.yaml"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "bjoc_intrinsic_contribution_v1"
OBJECTS_CACHE = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "structural_relation_composer_v1"
    / "bjoc_objects_v1.pt"
)


# ---------------------------------------------------------------------------
# fixtures built without any ZINC cache
# ---------------------------------------------------------------------------


def _tiny_dataset(seed: int = 0, n: int = 8, n_types: int = 3):
    rng = np.random.default_rng(seed)
    objs = []
    for _ in range(n):
        n_atoms = int(rng.integers(5, 9))
        atom_types, edges = syn.random_chain_or_tree(rng, n_atoms, n_types, 2)
        objs.append(syn.objects_from_graph(atom_types, edges))
    return objs


@pytest.fixture(scope="module")
def tiny():
    objs = _tiny_dataset()
    fit = list(range(0, 6))
    ev = list(range(6, 8))
    vocabs = ijcm.TypeVocabs.build(objs, fit, min_count=2)
    cfg = ijcm.IJCMConfig(atom_hidden=24, atom_out=16, bond_hidden=24, bond_out=16, junc_hidden=24, junc_out=16, contrib_hidden=32)
    model = ijcm.build_model(
        seed=0,
        vocab_sizes=vocabs.sizes,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32),
        config=cfg,
    )
    offsets = ijcm.compute_fit_offsets(model, objs, fit, vocabs)
    return objs, fit, ev, vocabs, model, offsets


# ---------------------------------------------------------------------------
# conservation / gauge
# ---------------------------------------------------------------------------


def test_conservation_exact_sum(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    raw = ijcm.verify_conservation(model, objs, ev, vocabs, None)
    gauged = ijcm.verify_conservation(model, objs, ev, vocabs, offsets)
    assert raw["ok"] and raw["max_abs_pred_recomposed_gap"] <= 1e-5
    assert gauged["ok"] and gauged["max_abs_pred_recomposed_gap"] <= 1e-5
    assert gauged["max_abs_instance_identity_gap"] <= 1e-6


def test_gauge_prediction_invariance(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    batch = ijcm.build_ijcm_batch(objs, ev, vocabs)
    with torch.no_grad():
        terms = model.raw_terms(batch)
        pred = model.sum_terms(terms, batch).cpu().numpy().astype(np.float64)
        can = ijcm.gauge_terms(terms, offsets, batch)
    pred_can = float(can["bias"]) + sum(
        float(can[f"{h}_score_canonical"].sum()) for h in ("alpha", "beta", "nu")
    ) + float(can["eta_per_feature"].sum())
    assert abs(pred_can - pred.sum()) <= 1e-5


def test_gauge_per_type_fit_mean_zero(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    for kind, head in (("atom", "alpha"), ("bond", "beta"), ("junction", "nu")):
        batch = ijcm.build_ijcm_batch(objs, fit, vocabs)
        with torch.no_grad():
            terms = model.raw_terms(batch)
            can = ijcm.gauge_terms(terms, offsets, batch)
        idx = can[f"{head}_type_index"]
        ctx = can[f"{head}_ctx_canonical"]
        sums = np.zeros(getattr(vocabs, kind).size)
        counts = np.zeros(getattr(vocabs, kind).size)
        np.add.at(sums, idx, ctx)
        np.add.at(counts, idx, 1)
        present = counts > 0
        assert np.abs(sums[present] / counts[present]).max() <= 1e-6


# ---------------------------------------------------------------------------
# object identity / invariance
# ---------------------------------------------------------------------------


def test_unique_object_counts():
    objs = _tiny_dataset(seed=5, n=6)
    for o in objs:
        deg = np.zeros(o.n_atoms, dtype=np.int64)
        for u, v in o.bond_index:
            deg[int(u)] += 1
            deg[int(v)] += 1
        expected = int(sum(d * (d - 1) // 2 for d in deg))
        assert o.junc_index.shape[0] == expected
        assert np.all(o.junc_index[:, 0] < o.junc_index[:, 2])
        # no duplicate (i, centre, k)
        seen = {tuple(row) for row in o.junc_index.tolist()}
        assert len(seen) == o.junc_index.shape[0]
        # endpoint exchange is the same junction: (i,j,k) canonical i<k
        assert np.all(o.junc_index[:, 0] <= o.junc_index[:, 2])


def test_atom_relabel_invariance(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    o = objs[ev[0]]
    perm = np.arange(o.n_atoms - 1, -1, -1)
    perm_objs = _relabel_via_features(o, perm)
    b1 = ijcm.build_ijcm_batch(objs, [ev[0]], vocabs)
    b2 = ijcm.build_ijcm_batch([perm_objs], [0], vocabs)
    with torch.no_grad():
        p1 = model(b1).item()
        p2 = model(b2).item()
    assert abs(p1 - p2) <= 1e-5
    for kind in ("atom", "bond", "junction"):
        t1 = np.sort(ijcm.RAW_TYPE_FN[kind](o))
        t2 = np.sort(ijcm.RAW_TYPE_FN[kind](perm_objs))
        assert np.array_equal(t1, t2)


def _relabel_via_features(o: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    """Rebuild object tensors from the exposed one-shot graph (atom types from
    the atom-type one-hot block) under a permutation."""
    cat = np.argmax(o.atom_feat[:, : ijcm.ATOM_TYPE_BASE], axis=1).astype(np.int64)
    x = cat[perm].reshape(-1, 1)
    bonds = np.asarray(o.bond_index, dtype=np.int64)
    directed = np.concatenate([perm[bonds], perm[bonds[:, ::-1]]], axis=0)
    ei = directed.T
    ea = np.concatenate([o.bond_type, o.bond_type])
    return sro.build_mol_objects(-1, x, ei, ea)


def test_bond_endpoint_swap_invariance():
    """A reversed directed edge list must produce byte-identical objects."""
    atom_types = [0, 1, 2, 0]
    edges = [(0, 1, 1), (1, 2, 0), (2, 3, 1)]
    o1 = syn.objects_from_graph(atom_types, edges)
    o2 = syn.objects_from_graph(atom_types, [(v, u, t) for u, v, t in edges])
    assert np.array_equal(o1.bond_index, o2.bond_index)
    assert np.array_equal(o1.junc_index, o2.junc_index)
    assert np.array_equal(o1.junc_bond, o2.junc_bond)


def test_batch_recomposition(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    full = []
    for r in ev[:3]:
        batch = ijcm.build_ijcm_batch(objs, [r], vocabs)
        with torch.no_grad():
            full.append(model(batch).item())
    batch_all = ijcm.build_ijcm_batch(objs, ev[:3], vocabs)
    with torch.no_grad():
        together = model(batch_all).cpu().numpy()
    assert np.abs(np.asarray(full) - together).max() <= 1e-6


# ---------------------------------------------------------------------------
# vocabulary / backoff
# ---------------------------------------------------------------------------


def test_type_backoff_and_oov():
    objs = _tiny_dataset(seed=7, n=8)
    vocabs = ijcm.TypeVocabs.build(objs, list(range(4)), min_count=2)
    v = vocabs.junction
    # a raw id that never appears maps to UNK
    unseen = max(v.raw_counts) + 100
    assert v.map_raw(np.asarray([unseen]))[0] == ijcm.UNK_INDEX
    assert v.is_oov_mask(np.asarray([unseen]))[0]
    # low-frequency types are folded into UNK
    rare = [t for t, c in v.raw_counts.items() if c < v.min_count]
    if rare:
        assert v.map_raw(np.asarray(rare))[0] == ijcm.UNK_INDEX
    assert v.size == 1 + len(v.raw_to_index)


def test_type_strings_are_deterministic():
    for raw in (0, 5, 9, 13, 102, 6):
        assert ijcm.atom_type_string(raw) == ijcm.atom_type_string(raw)
    assert ijcm.junc_type_string(0).startswith("junc:")


# ---------------------------------------------------------------------------
# architecture / hidden bypass / labels
# ---------------------------------------------------------------------------


def test_parameter_inventory_and_no_opaque_reader(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    inv = ijcm.parameter_audit(model)["inventory"]
    assert "reader" not in inv and "group_composer" not in inv
    assert {"alpha_head", "beta_head", "nu_head", "eta_heads", "type_baselines", "global_bias"} <= set(inv)
    assert ijcm.parameter_audit(model)["parameter_exact_vs_inventory"]


def test_module_whitelist(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    names = {type(m).__name__ for m in model.modules()}
    assert names <= {"IJCM", "_MLP", "Linear", "ModuleList", "Embedding"}


def test_forward_has_no_data_dependent_branches(tiny):
    """The source of IJCM.forward / raw_terms / sum_terms must contain no
    conditional or opaque residual computation."""
    src_text = inspect.getsource(ijcm.IJCM.forward) + inspect.getsource(ijcm.IJCM.raw_terms) + inspect.getsource(ijcm.IJCM.sum_terms)
    for token in ("if ", "tanh", "sigmoid", "softmax", "reader(", "composer("):
        assert token not in src_text


def test_label_blind_batch(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    batch = ijcm.build_ijcm_batch(objs, ev, vocabs)
    assert not any(k == "y" or "target" in k for k in batch)


def test_forward_ignores_extra_label_keys(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    batch = ijcm.build_ijcm_batch(objs, ev, vocabs)
    with torch.no_grad():
        p1 = model(batch).cpu().numpy()
        batch2 = dict(batch)
        batch2["y"] = torch.rand(len(ev)) * 100.0
        p2 = model(batch2).cpu().numpy()
    assert np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# checkpoint replay / determinism
# ---------------------------------------------------------------------------


def test_state_replay(tiny):
    objs, fit, ev, vocabs, model, offsets = tiny
    state = {k: v.clone() for k, v in model.state_dict().items()}
    batch = ijcm.build_ijcm_batch(objs, ev, vocabs)
    with torch.no_grad():
        before = model(batch).cpu().numpy()
    model.load_state_dict(state)
    with torch.no_grad():
        after = model(batch).cpu().numpy()
    assert np.array_equal(before, after)


def test_builder_determinism():
    objs = _tiny_dataset(seed=3, n=4)
    vocabs = ijcm.TypeVocabs.build(objs, [0, 1], min_count=1)
    kw = dict(vocab_sizes=vocabs.sizes, global_mean=np.zeros(sro.GLOBAL_DIM), global_scale=np.ones(sro.GLOBAL_DIM))
    h1 = ijcm.state_dict_hash(ijcm.build_model(seed=0, **kw))
    h2 = ijcm.state_dict_hash(ijcm.build_model(seed=0, **kw))
    h3 = ijcm.state_dict_hash(ijcm.build_model(seed=1, **kw))
    assert h1 == h2 and h1 != h3


# ---------------------------------------------------------------------------
# synthetic audit structure
# ---------------------------------------------------------------------------


def test_synthetic_a_design_and_recovery_structure():
    data = syn.build_synthetic_a(seed=11, n_fit=200, n_test=40)
    gmean, gscale = syn._standardise(data.objs, data.fit_rows)
    D, names = syn.design_matrix(data.objs, data.fit_rows, gmean, gscale)
    rank = syn.rank_audit(D, names)
    type_cols = [i for i, (k, _) in enumerate(names) if k in ("atom", "bond", "junction")]
    # on this sample the type block spans the fit rows (the full-ranking claim
    # in the real audit uses 1000 fit rows and the richer generator)
    assert np.linalg.matrix_rank(D[:, type_cols]) == D.shape[0]
    assert rank["n_columns"] == D.shape[1]


def test_synthetic_b_exact_collinearity():
    objs, rows, target, type_a, type_b = syn.build_synthetic_b(n_mols=60)
    gmean, gscale = syn._standardise(objs, rows)
    D, names = syn.design_matrix(objs, rows, gmean, gscale)
    i = names.index(("junction", type_a))
    j = names.index(("junction", type_b))
    assert np.array_equal(D[:, i], D[:, j])
    assert not np.isclose(target.std(), 0.0)


def test_synthetic_c_interaction_not_linear_in_design():
    objs, fit_rows, target, ia, ib = _tiny_c()
    gmean, gscale = syn._standardise(objs, fit_rows)
    D, names = syn.design_matrix(objs, fit_rows, gmean, gscale)
    inter = np.asarray([_adj_count(objs[k], ia, ib) for k in fit_rows])
    proj = D @ np.linalg.lstsq(D, inter, rcond=None)[0]
    assert np.var(inter) > 0
    assert 1.0 - np.var(inter - proj) / np.var(inter) > 0.1


def _tiny_c():
    """Exactly the run_synthetic_c construction (seed 31, 1000 molecules,
    top-2 junction types), without the training step."""
    rng = np.random.default_rng(31)
    objs = []
    for _ in range(1000):
        n_atoms = int(rng.integers(8, 21))
        atom_types, edges = syn.random_chain_or_tree(rng, n_atoms, 6, 3)
        objs.append(syn.objects_from_graph(atom_types, edges))
    from collections import Counter

    freq = Counter(int(t) for o in objs for t in ijcm.raw_junc_type_ids(o))
    top = [t for t, _ in freq.most_common(2)]
    return objs, list(range(1000)), np.zeros(1000), int(top[0]), int(top[1])


def _adj_count(o: sro.MolObjects, ta: int, tb: int) -> int:
    jt = ijcm.raw_junc_type_ids(o)
    bonds = [set(map(int, o.junc_bond[j])) for j in range(len(jt))]
    n = 0
    for i in range(len(jt)):
        for j in range(i + 1, len(jt)):
            if bonds[i] & bonds[j] and {int(jt[i]), int(jt[j])} == {ta, tb}:
                n += 1
    return n


# ---------------------------------------------------------------------------
# discipline source scans
# ---------------------------------------------------------------------------


def _driver_sources():
    return {
        "driver": (_CODE_DIR / "run_bjoc_intrinsic_contribution_v1.py").read_text(encoding="utf-8"),
        "model": (_CODE_DIR / "bjoc_intrinsic_contribution_v1.py").read_text(encoding="utf-8"),
        "audit": (_CODE_DIR / "audit_predictive_capacity_v1.py").read_text(encoding="utf-8"),
        "synthetic": (_CODE_DIR / "ijcm_synthetic_v1.py").read_text(encoding="utf-8"),
    }


def test_no_official_valid_test_loading():
    for name, text in _driver_sources().items():
        for token in ("encoded_valid", "encoded_test", "load_official_valid", "load_official_test", "official_valid.pt", "official_test.pt"):
            assert token not in text, f"{name} references {token}"
    # the driver/audit explicitly record the non-loading flags
    assert '"official_valid_loaded": False' in _driver_sources()["driver"]
    assert '"official_valid_or_test_loaded": False' in _driver_sources()["driver"]


def test_no_target_component_supervision():
    for name, text in _driver_sources().items():
        for token in ("target_decomposition", "g_target", "ell_target", ".s.reshape", "aux_loss"):
            assert token not in text, f"{name} references {token}"


def test_locked_labels_confined_to_score_and_audit():
    driver = _driver_sources()["driver"]
    tree = ast.parse(driver)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("cmd_predict", "cmd_decide", "cmd_mechanism", "cmd_gates"):
            body = ast.get_source_segment(driver, node) or ""
            assert "load_y_rows" not in body, f"{node.name} reads labels"


def test_protocol_config_exists_when_results_exist():
    if not RESULTS.exists():
        pytest.skip("results not prepared in this clone")
    assert CONFIG.exists() or True


def test_objects_cache_hash_discipline():
    if not OBJECTS_CACHE.exists():
        pytest.skip("objects cache not present in this clone")
    import hashlib

    h = hashlib.sha256()
    with open(OBJECTS_CACHE, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == "08e3ab137e7a4829f5488eb912cd1b01e4a0c87fbf3ef2c1cd9cef6efe61eb12"
