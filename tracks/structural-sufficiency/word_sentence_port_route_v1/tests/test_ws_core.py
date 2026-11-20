"""Fast CPU-only checks for the frozen word/sentence contracts.

These are contract tests (canonicalization, R, ERASED, merge learning, FULL vs
ERASED identical initial predictions), not a re-implementation of the analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import ws_common as C  # noqa: E402
import ws_words as W  # noqa: E402


def test_canonicalization_is_exact_and_relabel_invariant():
    labels = np.asarray([[6, 0, 0], [6, 0, 0], [8, 0, 0]], dtype=np.int64)
    bu = np.asarray([0, 1])
    bv = np.asarray([1, 2])
    bt = np.asarray([0, 1])
    canon, order = W.canonicalize(labels, bu, bv, bt)
    perm = np.asarray([2, 0, 1])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(3)
    canon2, _ = W.canonicalize(labels[perm], inv[bu], inv[bv], bt)
    assert canon == canon2
    # a different bond type must not collide
    canon3, _ = W.canonicalize(labels, bu, bv, np.asarray([0, 0]))
    assert canon3 != canon


def test_permutation_of_labels_only_is_not_an_isomorphism():
    labels = np.asarray([[6, 0, 0], [8, 0, 0]], dtype=np.int64)
    swapped = np.asarray([[8, 0, 0], [6, 0, 0]], dtype=np.int64)
    bu, bv, bt = np.asarray([0]), np.asarray([1]), np.asarray([0])
    canon_a, _ = W.canonicalize(labels, bu, bv, bt)
    canon_b, _ = W.canonicalize(swapped, bu, bv, bt)
    assert canon_a == canon_b  # relabelling: isomorphic


def test_template_identity_and_r_descriptor():
    table = W.TemplateTable()
    labels = np.asarray([[6, 0, 0]] * 6, dtype=np.int64)
    ring = [(i, (i + 1) % 6, 0) for i in range(6)]
    tid = table.intern(labels, np.asarray([b[0] for b in ring]),
                       np.asarray([b[1] for b in ring]), np.asarray([b[2] for b in ring]))
    tmpl = table.templates[tid]
    assert tmpl.n == 6
    # canonical positions are not the input order: pick one site per distance
    one = int(np.nonzero(tmpl.dist[0] == 1)[0][0])
    two = int(np.nonzero(tmpl.dist[0] == 2)[0][0])
    three = int(np.nonzero(tmpl.dist[0] == 3)[0][0])
    r1 = W.r_descriptor(tmpl, 0, one)
    r2 = W.r_descriptor(tmpl, 0, two)
    r3 = W.r_descriptor(tmpl, 0, three)
    assert not np.allclose(r1, r2) and not np.allclose(r2, r3) and not np.allclose(r1, r3)
    assert np.allclose(r1, W.r_descriptor(tmpl, one, 0))  # symmetry
    assert r1[0] == 0.0 and W.r_descriptor(tmpl, 0, 0)[0] == 1.0


def test_erased_table_is_label_compatible_mean():
    table = W.TemplateTable()
    labels = np.asarray([[6, 0, 0], [6, 0, 0], [8, 0, 0]], dtype=np.int64)
    bu = np.asarray([0, 1])
    bv = np.asarray([1, 2])
    bt = np.asarray([0, 0])
    tid = table.intern(labels, bu, bv, bt)
    tmpl = table.templates[tid]
    key = (W.label_code(labels[0]), W.label_code(labels[1]))
    assert key in tmpl.erased
    keys = {W.label_code(labels[p]) for p in range(3)}
    assert len(keys) == 2


def test_merge_learning_covers_each_atom_once():
    # tiny synthetic corpus: ethane-like chains sharing one two-atom word
    rows = [[6, 0, 1, 0, 3, 0, 4, 0, 0], [6, 0, 2, 0, 2, 0, 4, 0, 0], [8, 0, 1, 0, 1, 0, 4, 0, 0]]
    molecules = []
    from ws_probes import synthetic_molecule

    for g in range(6):
        molecules.append(synthetic_molecule(g, rows, [(0, 1, 0), (1, 2, 0)]))
    table = W.TemplateTable()
    history, stats = W.learn_merge_history(molecules, list(range(6)), table, min_df=3,
                                           max_multi_words=8)
    assert stats["rounds"] == len(history)
    states = W.replay_history(molecules, list(range(6)), table, history)
    W.verify_partition(molecules, list(range(6)), states)
    assert all(s.n > 1 for s in (table.templates[t] for t in history))


def test_full_and_erased_start_from_identical_predictions():
    torch = pytest.importorskip("torch")
    import ws_model as M
    import ws_train as T

    payload, freeze = T.load_prepared("ogbg-molesol")
    splits = {s: T.split_data(payload, "word", s) for s in ("train", "valid")}
    full = T.make_model(M.ModelSpec(encoder="code", mode="full"), payload, seed=0, device="cpu")
    erased = T.make_model(M.ModelSpec(encoder="code", mode="erased"), payload, seed=0,
                          device="cpu")
    full.eval()
    erased.eval()
    idx = np.arange(16)
    batch = M.to_device(M.pack_batch(splits["valid"], idx), "cpu")
    with torch.no_grad():
        a = full(batch)["pred"]
        b = erased(batch)["pred"]
    assert torch.allclose(a, b, atol=1e-6)
    assert not np.allclose(batch["pair_R"].numpy(), batch["pair_ER"].numpy())


def test_no_test_loader_is_built():
    source = (CODE / "ws_train.py").read_text(encoding="utf-8")
    assert '"test"' not in source.replace("terminal development-Test", "")
