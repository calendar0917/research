"""Fast correctness tests for cscl-bjoc-strc-predictive-feasibility-v1.

No ZINC data, no checkpoints, no GPU.  Covers the frozen mechanism contract:
the vocabulary rule, the shared-count parameterisation, the zero-init identity
with the base model, the gradient-sharing identities and the parameter budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_strc_v1 as strc  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

ZERO_MEAN = np.zeros(sro.GLOBAL_DIM, dtype=np.float32)
ONE_SCALE = np.ones(sro.GLOBAL_DIM, dtype=np.float32)


def _mol(atom_types, edges):
    n = len(atom_types)
    x = np.asarray(atom_types, dtype=np.int64).reshape(n, 1)
    ei, ea = [], []
    for u, v, bt in edges:
        ei.extend([(u, v), (v, u)])
        ea.extend([bt, bt])
    edge_index = np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0), dtype=np.int64)
    return sro.build_mol_objects(-1, x, edge_index, np.asarray(ea, dtype=np.int64))


def _tiny_objects():
    # a chain with a branch so that junctions exist
    return [
        _mol([0, 1, 0, 2], [(0, 1, 0), (1, 2, 1), (1, 3, 0)]),
        _mol([1, 1, 3, 0, 1], [(0, 1, 0), (1, 2, 0), (1, 3, 1), (3, 4, 0)]),
        _mol([0, 0, 1], [(0, 1, 0), (1, 2, 0)]),
    ]


def _vocab(objs=None):
    if objs is None:
        objs = _tiny_objects()
    return strc.build_context_vocabulary(objs)


def _batch(objs):
    return sro.batch_objects(objs)


def _base(seed=0):
    return src.build_bjoc(src.BJOCConfig(), seed=seed, organization_mode=src.NULL, global_mean=ZERO_MEAN, global_scale=ONE_SCALE, device="cpu")


def test_context_key_sorts_endpoints():
    assert strc.junction_context(3, 5, 1) == (1, 5, 3)
    assert strc.junction_context(1, 5, 3) == (1, 5, 3)
    assert strc.junction_context(3, 5, 3) == (3, 5, 3)


def test_vocabulary_rule_counts_once_and_orders():
    objs = _tiny_objects()
    vocab = _vocab(objs)
    # every junction in every molecule appears exactly once in counts_full
    total_junctions = sum(int(o.junc_index.shape[0]) for o in objs)
    assert sum(vocab["counts_full"].values()) == total_junctions
    assert vocab["covered_junctions"] + vocab["other_junctions"] == total_junctions
    # ordering: counts non-increasing, ties lexicographic
    counts = vocab["vocab_counts"]
    assert counts == sorted(counts, reverse=True)
    for i in range(len(vocab["vocab"]) - 1):
        if counts[i] == counts[i + 1]:
            assert tuple(vocab["vocab"][i]) <= tuple(vocab["vocab"][i + 1])


def test_vocabulary_tie_break_is_lexicographic():
    # two contexts with identical counts must sort by lexicographic context
    objs = [
        _mol([2, 1, 3], [(0, 1, 0), (1, 2, 0)]),  # context (2,1,3)
        _mol([1, 1, 2], [(0, 1, 0), (1, 2, 0)]),  # context (1,1,2)
    ]
    vocab = _vocab(objs)
    assert [tuple(c) for c in vocab["vocab"]] == [(1, 1, 2), (2, 1, 3)]
    assert vocab["vocab_counts"] == [1, 1]


def test_vocabulary_hash_determinism_and_lookup():
    vocab_a = _vocab()
    vocab_b = _vocab()
    assert vocab_a["vocab_sha256"] == vocab_b["vocab_sha256"] == strc.vocabulary_sha256(vocab_a)
    table = strc.vocab_lookup_table(vocab_a)
    for slot, ctx in enumerate(vocab_a["vocab"]):
        lo, centre, hi = ctx
        assert table[strc.context_key_index(lo, centre, hi)] == slot
    # an unseen context falls back to OTHER
    seen = {tuple(c) for c in vocab_a["vocab"]}
    unseen = next(
        (lo, c, hi)
        for lo in range(strc.ATOM_VOCAB)
        for c in range(strc.ATOM_VOCAB)
        for hi in range(strc.ATOM_VOCAB)
        if (lo, c, hi) not in seen
    )
    assert table[strc.context_key_index(*unseen)] == strc.OTHER_SLOT


def test_parameter_budget():
    vocab = _vocab()
    model = strc.wrap_strc(_base(), vocab, "cpu")
    audit = strc.strc_parameter_audit(model)
    assert audit["base_parameters"] == 413505
    assert audit["alpha_parameters"] == 28
    assert audit["beta_parameters"] == 129
    assert audit["new_parameters"] == 157
    assert audit["actual_parameters"] == 413662
    assert audit["parameter_exact"]


def test_zero_init_equals_base_exactly():
    vocab = _vocab()
    base = _base()
    model = strc.wrap_strc(base, vocab, "cpu")
    objs = _tiny_objects()
    batch = _batch(objs)
    with torch.no_grad():
        pred_strc = model(batch)
        pred_base = base(batch)
        residual = model.residual(batch)
    assert float(residual.abs().max()) == 0.0
    assert torch.equal(pred_strc, pred_base)


def test_count_identity_and_linearity():
    vocab = _vocab()
    model = strc.wrap_strc(_base(), vocab, "cpu")
    objs = _tiny_objects()
    batch = _batch(objs)
    a_counts, c_counts = model.residual.count_vectors(batch)
    assert a_counts.sum(1).tolist() == [o.n_atoms for o in objs]
    assert c_counts.sum(1).tolist() == [int(o.junc_index.shape[0]) for o in objs]
    # alpha = e_t * v must yield residual s_a * n_t * v for every molecule
    t = int(batch["atom_feat"][:, : strc.ATOM_VOCAB].argmax(1)[0])
    with torch.no_grad():
        model.residual.alpha.zero_()
        model.residual.alpha[t] = 2.5
        model.residual.beta.zero_()
        residual = model.residual(batch)
    expect = model.residual.s_a * a_counts[:, t] * 2.5
    assert torch.allclose(residual, expect, atol=1e-6)


def test_gradient_sharing_identity_single_molecule():
    vocab = _vocab()
    model = strc.wrap_strc(_base(), vocab, "cpu")
    obj = _tiny_objects()[0]
    batch = _batch([obj])
    model.zero_grad()
    pred = model(batch)[0]
    pred.backward()
    a_counts, c_counts = model.residual.count_vectors(batch)
    alpha_expect = model.residual.s_a * a_counts[0].detach()
    beta_expect = model.residual.s_j * c_counts[0].detach()
    assert torch.allclose(model.residual.alpha.grad, alpha_expect, atol=1e-6)
    assert torch.allclose(model.residual.beta.grad, beta_expect, atol=1e-6)


def test_batch_composition_consistency():
    vocab = _vocab()
    model = strc.wrap_strc(_base(), vocab, "cpu")
    objs = _tiny_objects()
    with torch.no_grad():
        single = model(_batch([objs[0]]))[0]
        in_batch = model(_batch(objs))[0]
    assert abs(float(single - in_batch)) < 1e-6


def test_gradient_activity_and_optimizer_membership():
    vocab = _vocab()
    model = strc.wrap_strc(_base(), vocab, "cpu")
    batch = _batch(_tiny_objects())
    y = torch.tensor([1.0, -2.0, 0.5])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    pred = model(batch)
    loss = F.l1_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    assert (model.residual.alpha.grad != 0).any()
    assert (model.residual.beta.grad != 0).any()
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert id(model.residual.alpha) in opt_ids and id(model.residual.beta) in opt_ids
    optimizer.step()
    assert (model.residual.alpha.detach() != 0).any() or (model.residual.beta.detach() != 0).any()


def test_no_forbidden_modules():
    vocab = _vocab()
    model = strc.wrap_strc(_base(), vocab, "cpu")
    assert src.forbidden_module_scan(model) == []
    text = (_CODE_DIR / "bjoc_strc_v1.py").read_text(encoding="utf-8")
    for token in ("MessagePassing", "Transformer", "MultiheadAttention", "GRU", "LSTM"):
        assert token not in text


def test_checkpoint_round_trip():
    vocab = _vocab()
    model = strc.wrap_strc(_base(), vocab, "cpu")
    with torch.no_grad():
        model.residual.alpha.normal_(generator=torch.Generator().manual_seed(1))
        model.residual.beta.normal_(generator=torch.Generator().manual_seed(2))
    batch = _batch(_tiny_objects())
    with torch.no_grad():
        before = model(batch)
    state = model.state_dict()
    assert "residual.alpha" in state and "residual.beta" in state
    # the non-persistent lookup buffer is absent from the state dict
    assert "residual.ctx_lookup" not in state
    model2 = strc.wrap_strc(_base(), vocab, "cpu")
    model2.load_state_dict(state)
    with torch.no_grad():
        after = model2(batch)
    assert torch.equal(before, after)


def test_config_frozen_block_matches_code():
    import json

    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "tracks/structure-contribution/configs/bjoc_strc_predictive_feasibility_v1.yaml").read_text(encoding="utf-8"))
    frozen = cfg["frozen"]
    assert frozen["context_keep"] == strc.CONTEXT_KEEP
    assert frozen["beta_size"] == strc.BETA_SIZE
    assert frozen["new_parameters"] == strc.NEW_PARAMETERS
    assert frozen["total_parameters"] == strc.TOTAL_PARAMETERS
    assert json.dumps(cfg)  # config is plain JSON-serialisable data
