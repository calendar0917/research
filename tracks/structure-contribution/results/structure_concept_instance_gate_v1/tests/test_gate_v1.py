"""Unit tests for ``structure_concept_instance_gate_v1``.

Fresh-clone safe: synthetic graphs only, CPU, no data/ dependency, no
checkpoints.  Run with:

    uv run pytest -q tracks/structure-contribution/results/structure_concept_instance_gate_v1/tests

The same tests are re-exported by
``tracks/structure-contribution/tests/test_structure_concept_instance_gate_v1.py``
so the repository-wide pytest configuration collects them too.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import analysis as A  # noqa: E402
import pipeline as P  # noqa: E402
import synthetic as S  # noqa: E402
from models import ARMS, build_model  # noqa: E402


# ---------------------------------------------------------------------------
# generator / preflight
# ---------------------------------------------------------------------------


def test_preflight_passes():
    rep = S.run_preflight()
    assert rep["checks"]["binding_counterexample"]["same_edges"]
    assert rep["checks"]["relation_counterexample"]["identical_attributed_patch_bag"]
    assert rep["checks"]["relabeling_invariance"]["patch_bags_equal"]
    assert rep["checks"]["instance_truth"]["per_graph_labels_match_rule"]


def test_binding_pair_marginals_identical_joint_bag_differs():
    graphs = S.all_graphs()
    g = next(x for x in graphs if x.family == "binding")
    twin = next(
        x
        for x in graphs
        if x.family == "binding"
        and x.base_id == g.base_id
        and x.meta["aux"] == g.meta["aux"]
        and x.label != g.label
    )
    assert g.edges == twin.edges
    assert S.attr_histogram(g) == S.attr_histogram(twin)
    assert S.unattributed_graph_key(g) == S.unattributed_graph_key(twin)
    assert S.patch_bag(g) != S.patch_bag(twin)
    # topology-only patch bag (all attributes collapsed) is identical
    flat_g = S.Graph("g", "binding", "b", "p", None, "train", "r", g.n, tuple(S.ATTR_C for _ in g.attrs), g.edges, 1, {})
    flat_t = S.Graph("t", "binding", "b", "p", None, "train", "r", twin.n, tuple(S.ATTR_C for _ in twin.attrs), twin.edges, -1, {})
    assert S.patch_bag(flat_g) == S.patch_bag(flat_t)


def test_relation_twins_have_identical_attributed_patch_bags():
    graphs = S.all_graphs()
    g = next(x for x in graphs if x.graph_id == "rel_cycle20_NO_short")
    twin = next(x for x in graphs if x.graph_id == g.bag_twin)
    assert g.edges == twin.edges
    assert S.attr_histogram(g) == S.attr_histogram(twin)
    assert S.patch_bag(g) == S.patch_bag(twin)
    assert S.patch_sap_bag(g) == S.patch_sap_bag(twin)
    assert g.label == 1 and twin.label == -1
    assert g.meta["d"] == 5 and twin.meta["d"] == 9


def test_composition_bag_twins():
    graphs = S.all_graphs()
    for g in graphs:
        if g.family != "composition" or not g.bag_twin:
            continue
        twin = next(x for x in graphs if x.graph_id == g.bag_twin)
        assert S.patch_bag(g) == S.patch_bag(twin)
        assert g.label != twin.label


def test_relabeling_preserves_patch_features_and_sap():
    g = next(x for x in S.all_graphs() if x.family == "relation")
    rng = np.random.default_rng(3)
    perm = rng.permutation(g.n)
    g2 = S.relabel_graph(g, perm)
    feats1 = {p.edge: (p.feature, p.sap) for p in S.extract_patches(g)}
    feats2 = {p.edge: (p.feature, p.sap) for p in S.extract_patches(g2)}
    for (u, v), (feat, sap) in feats1.items():
        mapped = tuple(sorted((int(perm[u]), int(perm[v]))))
        assert feats2[mapped][0] == feat
        assert feats2[mapped][1] == sap


def test_splits_are_base_disjoint_and_balanced():
    graphs = S.all_graphs()
    base_split: dict[str, set[str]] = {}
    for g in graphs:
        base_split.setdefault(g.base_id, set()).add(g.split)
    assert all(len(s) == 1 for s in base_split.values())
    for split in ("train", "dev", "test"):
        labels = [g.label for g in graphs if g.split == split]
        assert sum(1 for x in labels if x == 1) == sum(1 for x in labels if x == -1)


def test_matched_pair_marginals_identical():
    graphs = S.all_graphs()
    by_pair: dict[str, list] = {}
    for g in graphs:
        by_pair.setdefault(g.pair_id, []).append(g)
    for pair_id, gs in by_pair.items():
        if len(gs) < 2:
            continue
        fingerprints = {
            (len(x.edges), x.n, tuple(sorted(Counter(x.attrs).items())), len(S.extract_patches(x)))
            for x in gs
        }
        assert len(fingerprints) == 1, pair_id


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def _tensors(gid: str = "rel_cycle20_NO_short"):
    graph = next(x for x in S.all_graphs() if x.graph_id == gid)
    patches = S.extract_patches(graph)
    x = torch.tensor(np.stack([np.asarray(p.feature, dtype=np.float32) for p in patches]))
    dist = S.anchor_distance_matrix(graph, patches)
    rel = torch.tensor(S.relation_features(dist, patches))
    pair_ok = ~torch.eye(len(patches), dtype=torch.bool)
    return graph, patches, x, rel, pair_ok


@pytest.mark.parametrize("arm", ["C", "D"])
def test_hard_forward_is_concept_only(arm):
    _, _, x, rel, pair_ok = _tensors()
    model = build_model(arm, 0)
    logit, info = model(x, rel, pair_ok)
    ids = info["concept_ids"]
    hard = model.vq.prototypes[ids]
    assert torch.equal(info["hard"], hard)
    l2 = model.logits_from_concepts(ids, rel, pair_ok)
    assert torch.equal(logit.detach(), l2.detach())
    # reader input names are exactly the declared discrete tensors
    expected = {"C": {"concept_histogram", "concept_mean"}, "D": {"concept_histogram", "concept_mean", "pair_mean", "pair_max"}}[arm]
    assert {t["name"] for t in model.reader_input_table()} == expected
    # scaling the continuous encoder output (same argmax) does not change the reader
    logit2, info2 = model.forward_from_z(info["z"] * 0.25, rel, pair_ok)
    assert bool((info2["concept_ids"] == ids).all())
    assert torch.equal(logit.detach(), logit2.detach())


@pytest.mark.parametrize("arm", ["C", "D"])
def test_concept_gradients_flow(arm):
    _, _, x, rel, pair_ok = _tensors()
    model = build_model(arm, 0)
    y = torch.tensor(1.0)
    logit, info = model(x, rel, pair_ok)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
    loss = loss + torch.nn.functional.mse_loss(info["z"].detach(), info["hard"])
    loss.backward()
    enc = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
    proto = float(model.vq.prototypes.grad.abs().sum())
    assert enc > 0
    assert proto > 0


@pytest.mark.parametrize("arm", list(ARMS))
def test_patch_order_permutation_invariance(arm):
    _, _, x, rel, pair_ok = _tensors("rel_cycle20_NS_short")
    model = build_model(arm, 0)
    logit, _ = model(x, rel, pair_ok)
    rng = np.random.default_rng(0)
    perm = torch.tensor(rng.permutation(x.shape[0]))
    inv = torch.argsort(perm)
    xp = x[perm]
    relp = rel[perm][:, perm]
    pair_okp = pair_ok[perm][:, perm]
    logitp, _ = model(xp, relp, pair_okp)
    assert torch.allclose(logit, logitp, atol=1e-5)


@pytest.mark.parametrize("arm", list(ARMS))
def test_relation_diagonal_not_used(arm):
    _, _, x, rel, pair_ok = _tensors()
    model = build_model(arm, 0)
    logit, _ = model(x, rel, pair_ok)
    rel2 = rel.clone()
    idx = torch.arange(rel.shape[0])
    rel2[idx, idx] = 999.0
    logit2, _ = model(x, rel2, pair_ok)
    assert torch.allclose(logit, logit2, atol=1e-6)


def test_parameter_budget_matched():
    counts = {arm: build_model(arm, 0).n_parameters() for arm in ARMS}
    assert max(counts.values()) / min(counts.values()) <= 1.8
    assert all(v < 30_000 for v in counts.values())


def test_head_input_provenance_declares_no_continuous_for_concepts():
    for arm in ("C", "D"):
        model = build_model(arm, 0)
        _, _, x, rel, pair_ok = _tensors()
        model(x, rel, pair_ok)
        prov = " ".join(t["provenance"] for t in model.reader_input_table())
        assert "encoder hidden state" not in prov
        assert "hard" in prov


# ---------------------------------------------------------------------------
# scramble control
# ---------------------------------------------------------------------------


def test_relation_scramble_is_label_blind_and_breaks_twin_equality():
    graphs = S.all_graphs()
    patches = {g.graph_id: S.extract_patches(g) for g in graphs}
    dist = {g.graph_id: S.anchor_distance_matrix(g, patches[g.graph_id]) for g in graphs}
    train_ids = [g.graph_id for g in graphs if g.split == "train"]
    scr = S.scrambled_relation_features(dist, patches, train_ids)
    g = next(x for x in graphs if x.graph_id == "rel_cycle20_NO_short")
    twin = next(x for x in graphs if x.graph_id == g.bag_twin)
    assert not np.array_equal(scr[g.graph_id], scr[twin.graph_id])
    # all off-diagonal entries come from the frozen train pool: value counts differ
    # from the true relation features on the same graph
    true_feat = S.relation_features(dist[g.graph_id], patches[g.graph_id])
    assert not np.array_equal(scr[g.graph_id], true_feat)


# ---------------------------------------------------------------------------
# pipeline smoke
# ---------------------------------------------------------------------------


def test_training_smoke_and_hard_soft_paths():
    data, _ = P.prepare_data()
    cfg = P.TrainConfig(max_epochs=2, patience=2, batch_size=32, lr=3e-3)
    for arm in ("A", "D"):
        run = P.train_arm(arm, 0, data, relation_source="true", cfg=cfg)
        model = run["model"]
        gid = "rel_cycle20_NO_short"
        logit, prob = P.predict_graph(model, data[gid], "true", "hard")
        assert 0.0 <= prob <= 1.0
        if arm == "D":
            logit_s, prob_s = P.predict_graph(model, data[gid], "true", "soft")
            assert 0.0 <= prob_s <= 1.0


def test_auc_style_metrics_shapes():
    rows = {
        "a": {"prob": 0.9, "logit": 2.0, "soft_prob": 0.8, "soft_logit": 1.4, "label": 1, "family": "relation", "bag_twin": "b", "pair_id": "p"},
        "b": {"prob": 0.1, "logit": -2.0, "soft_prob": 0.2, "soft_logit": -1.4, "label": -1, "family": "relation", "bag_twin": "a", "pair_id": "p"},
    }
    m = A.matched_pair_metrics(rows, "relation")
    assert m["n_pairs"] == 1
    assert m["pair_balanced_accuracy"] == 1.0
    lim = A.bag_limit_check(rows, "relation")
    assert lim["max_abs_logit_diff_within_pair"] == pytest.approx(4.0)
