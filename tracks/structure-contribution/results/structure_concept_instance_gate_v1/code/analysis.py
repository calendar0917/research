"""Metric computation, concept semantics, explanation recovery and gates.

All numbers written to ``metrics.json`` come from this module.  The module
distinguishes three evidence classes explicitly:

1. **mathematical facts** (identical patch bags, information-theoretic limits,
   exact hard-forward equalities) - asserted, not learned;
2. **learning results** (accuracy/leakage/concept alignment of the trained
   models) - measured;
3. **interpretation** (concept ablation sensitivity vs synthetic ground truth) -
   reported with its rank/percentile, never as causal chemistry.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
from sklearn.tree import DecisionTreeClassifier

from models import GateModel
from pipeline import GraphTensors, collect_concepts

FAMILY_ORDER = ("binding", "relation", "composition")


# ---------------------------------------------------------------------------
# basic prediction metrics
# ---------------------------------------------------------------------------


def balanced_accuracy(probs: Sequence[float], labels: Sequence[int]) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    yhat = (probs >= 0.5).astype(int)
    pos = labels == 1
    neg = ~pos
    acc_pos = float((yhat[pos] == 1).mean()) if pos.any() else float("nan")
    acc_neg = float((yhat[neg] == 0).mean()) if neg.any() else float("nan")
    return float(np.nanmean([acc_pos, acc_neg]))


def eval_metrics(rows: Mapping[str, dict]) -> dict:
    rows = list(rows.values())
    labels = [1 if r["label"] > 0 else 0 for r in rows]
    out: dict = {
        "n": len(rows),
        "balanced_accuracy": balanced_accuracy([r["prob"] for r in rows], labels),
        "mean_prob": float(np.mean([r["prob"] for r in rows])),
    }
    hard = np.asarray([r["prob"] for r in rows])
    soft = np.asarray([r["soft_prob"] for r in rows if r["soft_prob"] is not None], dtype=float)
    if len(soft) == len(hard):
        out["hard_soft"] = {
            "balanced_accuracy_hard": balanced_accuracy(hard, labels),
            "balanced_accuracy_soft": balanced_accuracy(soft, labels),
            "mean_abs_diff": float(np.mean(np.abs(hard - soft))),
            "max_abs_diff": float(np.max(np.abs(hard - soft))),
        }
    by_family: dict[str, dict] = {}
    for fam in FAMILY_ORDER:
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        lab = [1 if r["label"] > 0 else 0 for r in sub]
        by_family[fam] = {
            "n": len(sub),
            "balanced_accuracy": balanced_accuracy([r["prob"] for r in sub], lab),
            "mean_prob": float(np.mean([r["prob"] for r in sub])),
        }
    out["by_family"] = by_family
    return out


def matched_pair_metrics(rows: Mapping[str, dict], family: str | None = None) -> dict:
    """Matched bag-twin pairs: one positive and one negative graph with the
    *same* patch bag (relation / composition families)."""
    rows = {k: v for k, v in rows.items() if family is None or v["family"] == family}
    seen: set[tuple[str, str]] = set()
    pair_scores: list[float] = []
    identical = 0
    pair_details: list[dict] = []
    for gid, r in rows.items():
        twin = r["bag_twin"]
        if not twin or twin not in rows:
            continue
        key = tuple(sorted((gid, twin)))
        if key in seen:
            continue
        seen.add(key)
        a, b = rows[gid], rows[twin]
        if a["label"] == b["label"]:
            continue
        correct_a = int((a["prob"] >= 0.5) == (a["label"] > 0))
        correct_b = int((b["prob"] >= 0.5) == (b["label"] > 0))
        pair_scores.append((correct_a + correct_b) / 2.0)
        same_pred = abs(a["prob"] - b["prob"]) < 1e-9
        identical += int(same_pred)
        pair_details.append(
            {
                "pair": list(key),
                "family": a["family"],
                "pair_id": a["pair_id"],
                "probs": [a["prob"], b["prob"]],
                "labels": [a["label"], b["label"]],
                "pair_accuracy": (correct_a + correct_b) / 2.0,
                "identical_prediction": bool(same_pred),
            }
        )
    return {
        "n_pairs": len(pair_scores),
        "pair_balanced_accuracy": float(np.mean(pair_scores)) if pair_scores else float("nan"),
        "n_identical_predictions": identical,
        "details": pair_details,
    }


def hard_soft_gap(rows: Mapping[str, dict]) -> dict:
    hard = np.asarray([r["prob"] for r in rows.values()], dtype=float)
    soft = np.asarray([r["soft_prob"] for r in rows.values() if r["soft_prob"] is not None], dtype=float)
    labels = [1 if r["label"] > 0 else 0 for r in rows.values()]
    if len(soft) != len(hard):
        return {"available": False}
    return {
        "available": True,
        "hard_balanced_accuracy": balanced_accuracy(hard, labels),
        "soft_balanced_accuracy": balanced_accuracy(soft, labels),
        "gap_soft_minus_hard": balanced_accuracy(soft, labels) - balanced_accuracy(hard, labels),
        "mean_abs_diff": float(np.mean(np.abs(hard - soft))),
        "max_abs_diff": float(np.max(np.abs(hard - soft))),
    }


# ---------------------------------------------------------------------------
# information-theoretic limit (mathematical fact)
# ---------------------------------------------------------------------------


def bag_limit_check(rows: Mapping[str, dict], family: str | None = None) -> dict:
    """Identical-input pairs must have bit-identical predictions for A/C.

    Returns the max absolute logit difference across matched twins and whether
    any pair is both correctly classified with different predictions.
    """
    rows = {k: v for k, v in rows.items() if family is None or v["family"] == family}
    seen: set[tuple[str, str]] = set()
    max_diff = 0.0
    pair_acc: list[float] = []
    for gid, r in rows.items():
        twin = r["bag_twin"]
        if not twin or twin not in rows:
            continue
        key = tuple(sorted((gid, twin)))
        if key in seen:
            continue
        seen.add(key)
        a, b = rows[gid], rows[twin]
        max_diff = max(max_diff, abs(a["logit"] - b["logit"]))
        correct_a = int((a["prob"] >= 0.5) == (a["label"] > 0))
        correct_b = int((b["prob"] >= 0.5) == (b["label"] > 0))
        pair_acc.append((correct_a + correct_b) / 2.0)
    return {
        "n_pairs": len(pair_acc),
        "max_abs_logit_diff_within_pair": max_diff,
        "pair_balanced_accuracy": float(np.mean(pair_acc)) if pair_acc else float("nan"),
    }


# ---------------------------------------------------------------------------
# head input audit
# ---------------------------------------------------------------------------


def head_input_audit(model: GateModel, data: Mapping[str, GraphTensors], split: str = "test") -> dict:
    """Prove the reader is a function of (hard concept id, declared relations).

    For arms C/D:
    * full-forward logits must equal ``logits_from_concepts(idx)`` bit-exactly;
    * re-running the reader with a scaled encoder output (same argmax) must be
      bit-exact;
    * the recorded reader-input table must contain only the four declared
      discrete tensors.
    """
    arm = model.cfg.arm
    out: dict = {"arm": arm, "split": split, "n_graphs": 0}
    if arm in ("A", "B"):
        out["applicable"] = False
        out["note"] = "arms A/B intentionally read continuous patch embeddings"
        return out
    max_full_vs_concepts = 0.0
    max_full_vs_scaled = 0.0
    same_argmax = True
    tables: list[list[dict]] = []
    expected = {
        "C": {"concept_histogram", "concept_mean"},
        "D": {"concept_histogram", "concept_mean", "pair_mean", "pair_max"},
    }[arm]
    with __import__("torch").no_grad():
        for gid, d in sorted(data.items()):
            if d.split != split:
                continue
            out["n_graphs"] += 1
            logit, info = model(d.x, d.rel_true, d.pair_ok)
            ids = info["concept_ids"]
            l2 = model.logits_from_concepts(ids, d.rel_true, d.pair_ok)
            max_full_vs_concepts = max(max_full_vs_concepts, abs(float(logit) - float(l2)))
            z = info["z"]
            logit3, info3 = model.forward_from_z(z * 0.5, d.rel_true, d.pair_ok)
            max_full_vs_scaled = max(max_full_vs_scaled, abs(float(logit) - float(logit3)))
            same_argmax &= bool((info3["concept_ids"] == ids).all())
            tables.append(model.reader_input_table())
    out.update(
        {
            "applicable": True,
            "max_abs_diff_full_vs_concept_only": max_full_vs_concepts,
            "max_abs_diff_full_vs_scaled_encoder": max_full_vs_scaled,
            "scaled_encoder_preserves_argmax": bool(same_argmax),
            "exact_match": bool(max_full_vs_concepts == 0.0 and max_full_vs_scaled == 0.0),
            "reader_input_names": sorted({t["name"] for tt in tables for t in tt}),
            "reader_input_provenance": tables[0] if tables else [],
            "declared_names_match": bool(
                {t["name"] for tt in tables for t in tt} == expected
            ) if tables else False,
        }
    )
    return out


# ---------------------------------------------------------------------------
# concept semantics
# ---------------------------------------------------------------------------


def concept_records(data: Mapping[str, GraphTensors], model: GateModel) -> list[dict]:
    """One record per patch: learned concept, true SAP, family/split/position."""
    recs: list[dict] = []
    for gid, d in sorted(data.items()):
        ids = collect_concepts(model, d)
        for j, p in enumerate(d.patches):
            recs.append(
                {
                    "graph_id": gid,
                    "family": d.family,
                    "split": d.split,
                    "base_id": d.base_id,
                    "pair_id": d.pair_id,
                    "label": d.label,
                    "patch_index": j,
                    "concept": int(ids[j]),
                    "sap": p.sap,
                    "sap_desc": p.sap_desc,
                    "special": list(p.special),
                    "nodes": list(p.nodes),
                    "anchor_lo": int(min(p.edge)),
                    "anchor_hi": int(max(p.edge)),
                    "n_nodes": int(d.n_nodes),
                    "n_edges": int(d.n_edges),
                    "n_patches": len(d.patches),
                    "meta": dict(d.meta),
                }
            )
    return recs


def hungarian_mapping(concepts: Sequence[int], saps: Sequence[str]) -> tuple[float, dict[int, str], np.ndarray, list[int], list[str]]:
    cs = sorted(set(concepts))
    ss = sorted(set(saps))
    ci = {c: i for i, c in enumerate(cs)}
    si = {s: i for i, s in enumerate(ss)}
    M = np.zeros((len(cs), len(ss)))
    for c, s in zip(concepts, saps):
        M[ci[c], si[s]] += 1
    r, cidx = linear_sum_assignment(-M)
    matched = float(M[r, cidx].sum())
    mapping = {cs[i]: ss[j] for i, j in zip(r, cidx)}
    return matched / max(len(concepts), 1), mapping, M, cs, ss


def contingency_stats(records: Sequence[dict], split: str | None = None, min_sap_support: int = 1) -> dict:
    if min_sap_support > 1:
        support = Counter(r["sap"] for r in records)
        supported = {s for s, n in support.items() if n >= min_sap_support}
    else:
        supported = None
    sub = [
        r
        for r in records
        if (split is None or r["split"] == split) and (supported is None or r["sap"] in supported)
    ]
    concepts = [r["concept"] for r in sub]
    saps = [r["sap"] for r in sub]
    purity, mapping, M, cs, ss = hungarian_mapping(concepts, saps)
    return {
        "n_patches": len(sub),
        "n_concepts_used": len(cs),
        "n_saps": len(ss),
        "matched_purity": purity,
        "nmi": float(normalized_mutual_info_score(saps, concepts, average_method="arithmetic")),
        "concept_to_sap": {str(k): v for k, v in mapping.items()},
        "sap_support": {s: int(np.sum(M[:, si])) for si, s in enumerate(ss)},
    }


def key_sap_purity(records: Sequence[dict], key_saps: set[str], split: str | None = None) -> dict:
    sub = [r for r in records if (split is None or r["split"] == split) and r["sap"] in key_saps]
    if not sub:
        return {"n_patches": 0, "dominant_share": float("nan")}
    counts = Counter(r["concept"] for r in sub)
    total = sum(counts.values())
    dom_concept, dom_count = counts.most_common(1)[0]
    return {
        "n_patches": len(sub),
        "dominant_concept": int(dom_concept),
        "dominant_share": dom_count / total,
        "n_concepts": len(counts),
    }


def cross_base_consistency(records: Sequence[dict], min_patches: int = 5) -> dict:
    """For SAPs shared by >= 2 base graphs, does the same concept dominate?"""
    by_sap: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_sap[r["sap"]].append(r)
    shares: list[float] = []
    n_saps = 0
    for sap, rs in by_sap.items():
        if len(rs) < min_patches:
            continue
        per_base: dict[str, Counter] = defaultdict(Counter)
        for r in rs:
            per_base[r["base_id"]][r["concept"]] += 1
        if len(per_base) < 2:
            continue
        global_dom = Counter(r["concept"] for r in rs).most_common(1)[0][0]
        agree = sum(1 for c in per_base.values() if c.most_common(1)[0][0] == global_dom)
        shares.append(agree / len(per_base))
        n_saps += 1
    return {
        "n_saps": n_saps,
        "mean_base_agreement": float(np.mean(shares)) if shares else float("nan"),
        "min_base_agreement": float(np.min(shares)) if shares else float("nan"),
    }


def cross_seed_agreement(records0: Sequence[dict], records1: Sequence[dict]) -> dict:
    assert len(records0) == len(records1)
    keys0 = [(r["graph_id"], r["patch_index"]) for r in records0]
    keys1 = [(r["graph_id"], r["patch_index"]) for r in records1]
    assert keys0 == keys1
    c1 = [r["concept"] for r in records0]
    c2 = [r["concept"] for r in records1]
    cs1 = sorted(set(c1))
    cs2 = sorted(set(c2))
    i1 = {c: i for i, c in enumerate(cs1)}
    i2 = {c: i for i, c in enumerate(cs2)}
    M = np.zeros((len(cs1), len(cs2)))
    for a, b in zip(c1, c2):
        M[i1[a], i2[b]] += 1
    r, cidx = linear_sum_assignment(-M)
    matched = float(M[r, cidx].sum())
    sap0 = [r_["sap"] for r_ in records0]
    sap1 = [r_["sap"] for r_ in records1]
    p0, _, _, _, _ = hungarian_mapping(c1, sap0)
    p1, _, _, _, _ = hungarian_mapping(c2, sap1)
    return {
        "n_patches": len(c1),
        "concept_assignment_agreement": matched / len(c1),
        "matched_purity_seed0": p0,
        "matched_purity_seed1": p1,
    }


def degeneracy_probes(records: Sequence[dict]) -> dict:
    concepts = np.asarray([r["concept"] for r in records])
    labels = np.asarray([1 if r["label"] > 0 else 0 for r in records])
    fams = sorted({r["family"] for r in records})
    x_rows = []
    for r in records:
        fam = [1.0 if r["family"] == f else 0.0 for f in fams]
        x_rows.append(
            [
                r["n_nodes"] / 20.0,
                r["n_edges"] / 30.0,
                r["n_patches"] / 30.0,
                r["anchor_lo"] / max(r["n_edges"], 1),
                r["anchor_hi"] / max(r["n_edges"], 1),
                *fam,
            ]
        )
    X = np.asarray(x_rows)
    g_rows = [[r["n_nodes"], r["n_edges"], r["n_patches"], *[1.0 if r["family"] == f else 0.0 for f in fams]] for r in records]
    Xg = np.asarray(g_rows)
    train = np.asarray([r["split"] == "train" for r in records])
    test = np.asarray([r["split"] == "test" for r in records])
    use_holdout = bool(train.any() and test.any())
    Xfit, yfit = (X[train], concepts[train]) if use_holdout else (X, concepts)
    Xev, yev = (X[test], concepts[test]) if use_holdout else (X, concepts)
    clf = LogisticRegression(max_iter=300, C=1.0)
    clf.fit(Xfit, yfit)
    probe_acc = float(clf.score(Xev, yev))
    ref = concepts[train] if use_holdout else concepts
    majority = float(Counter(ref.tolist()).most_common(1)[0][1] / len(ref))
    base_rate = Counter(yev.tolist()).most_common(1)[0][1] / len(yev)
    Xgfit, ygfit = (Xg[train], labels[train]) if use_holdout else (Xg, labels)
    Xgev, ygev = (Xg[test], labels[test]) if use_holdout else (Xg, labels)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(Xgfit, ygfit)
    label_probe = float(tree.score(Xgev, ygev))
    return {
        "concept_from_marginals_accuracy": probe_acc,
        "concept_majority_baseline": majority,
        "label_from_graph_marginals_accuracy": label_probe,
        "label_majority_baseline": float(base_rate),
        "holdout_train_to_test": use_holdout,
        "n": len(records),
    }


def conditional_mi(records: Sequence[dict]) -> dict:
    concepts = [r["concept"] for r in records]
    saps = [r["sap"] for r in records]
    labels = [1 if r["label"] > 0 else 0 for r in records]
    i_cy = mutual_info_score(concepts, labels)
    i_sy = mutual_info_score(saps, labels)
    total = len(records)
    by_sap: dict[str, list[int]] = defaultdict(list)
    for r in records:
        by_sap[r["sap"]].append(r)
    cond = 0.0
    for sap, rs in by_sap.items():
        w = len(rs) / total
        cond += w * mutual_info_score([x["concept"] for x in rs], [1 if x["label"] > 0 else 0 for x in rs])
    return {
        "I_concept_label": float(i_cy),
        "I_sap_label": float(i_sy),
        "I_concept_label_given_sap": float(cond),
    }


def concept_examples(records: Sequence[dict], top_n: int = 3) -> list[dict]:
    by_concept: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_concept[r["concept"]].append(r)
    out = []
    for k in sorted(by_concept):
        rs = by_concept[k]
        sap_counts = Counter(r["sap"] for r in rs)
        example = max(rs, key=lambda r: sap_counts[r["sap"]])
        out.append(
            {
                "concept": k,
                "n": len(rs),
                "top_saps": [
                    {"sap": s, "desc": next(r["sap_desc"] for r in rs if r["sap"] == s), "count": c}
                    for s, c in sap_counts.most_common(top_n)
                ],
                "families": dict(Counter(r["family"] for r in rs)),
                "example_graph": example["graph_id"],
                "example_edge": [example["anchor_lo"], example["anchor_hi"]],
                "example_nodes": example["nodes"],
                "example_special": example["special"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# explanation recovery (model-level ablations vs generator-level truth)
# ---------------------------------------------------------------------------


def concept_ablation_sensitivity(
    model: GateModel,
    data: Mapping[str, GraphTensors],
    family: str,
    split: str = "test",
) -> dict[int, float]:
    import torch

    sens: dict[int, list[float]] = defaultdict(list)
    with torch.no_grad():
        for gid, d in sorted(data.items()):
            if d.split != split or d.family != family:
                continue
            ids = collect_concepts(model, d)
            full = model.logits_from_concepts(torch.tensor(ids), d.rel_true, d.pair_ok)
            for k in sorted(set(int(x) for x in ids)):
                keep = torch.tensor(ids != k)
                ab = model.logits_from_concepts(torch.tensor(ids), d.rel_true, d.pair_ok, extra_patch_mask=keep)
                sens[k].append(abs(float(ab) - float(full)))
    return {k: float(np.mean(v)) for k, v in sens.items()}


def pair_ablation_sensitivity(
    model: GateModel,
    data: Mapping[str, GraphTensors],
    family: str,
    split: str = "test",
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    import torch

    sens: dict[tuple[int, int], list[float]] = defaultdict(list)
    with torch.no_grad():
        for gid, d in sorted(data.items()):
            if d.split != split or d.family != family:
                continue
            ids = collect_concepts(model, d)
            ids_t = torch.tensor(ids)
            full = model.logits_from_concepts(ids_t, d.rel_true, d.pair_ok)
            present = sorted(set(int(x) for x in ids))
            for k1 in present:
                for k2 in present:
                    allowed = ~((ids_t[:, None] == k1) & (ids_t[None, :] == k2))
                    ab = model.logits_from_concepts(ids_t, d.rel_true, d.pair_ok, pair_allowed=allowed)
                    sens[(k1, k2)].append(abs(float(ab) - float(full)))
    out = {k: float(np.mean(v)) for k, v in sens.items()}
    support = {k: len(v) for k, v in sens.items()}
    return out, support


def binding_recovery(
    model: GateModel,
    data: Mapping[str, GraphTensors],
    records: Sequence[dict],
    split: str = "test",
) -> dict:
    # ground truth key SAPs: most common SAP of patches containing the decisive
    # N atom, separately for positive (N in triangle) and negative graphs
    key_saps: dict[str, str] = {}
    for lab, tag in ((1, "pos"), (-1, "neg")):
        counts: Counter = Counter()
        for gid, d in data.items():
            if d.family != "binding" or d.label != lab:
                continue
            n_pos = d.meta.get("p_in") if lab > 0 else d.meta.get("p_out")
            for p in d.patches:
                if n_pos in p.nodes:
                    counts[p.sap] += 1
        if counts:
            key_saps[tag] = counts.most_common(1)[0][0]
    key_set = set(key_saps.values())
    sens = concept_ablation_sensitivity(model, data, "binding", split)
    key_mass: dict[int, float] = defaultdict(float)
    n_key = 0
    for r in records:
        if r["family"] != "binding" or r["split"] != split:
            continue
        if r["sap"] in key_set:
            key_mass[r["concept"]] += 1.0
            n_key += 1
    concepts = sorted(set(sens) | set(key_mass))
    x = [key_mass.get(k, 0.0) / max(n_key, 1) for k in concepts]
    y = [sens.get(k, 0.0) for k in concepts]
    rho = float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else float("nan")
    top_key_concept = max(concepts, key=lambda k: key_mass.get(k, 0.0)) if concepts else None
    rank = None
    if top_key_concept is not None and len(y) > 0:
        order = sorted(concepts, key=lambda k: -sens.get(k, 0.0))
        rank = order.index(top_key_concept) + 1
    return {
        "key_saps": key_saps,
        "n_key_patches": n_key,
        "key_mass_spearman_vs_ablation_sensitivity": rho,
        "top_key_concept": top_key_concept,
        "top_key_concept_sensitivity_rank": rank,
        "n_concepts": len(concepts),
        "sensitivity": {str(k): sens.get(k, 0.0) for k in concepts},
        "key_mass": {str(k): key_mass.get(k, 0.0) / max(n_key, 1) for k in concepts},
    }


def relation_recovery(
    model: GateModel,
    data: Mapping[str, GraphTensors],
    records: Sequence[dict],
    split: str = "test",
    pair_support_min: int = 3,
) -> dict:
    # ground truth SAPs of patches containing the two marked heteroatoms, on the
    # N-O subfamily (the only relation subfamily the binding rule also uses)
    sap_n_counts: Counter = Counter()
    sap_o_counts: Counter = Counter()
    for gid, d in data.items():
        if d.family != "relation" or d.meta.get("roles") != ["N", "O"]:
            continue
        nodes = d.meta.get("mark_nodes", [0, int(d.meta.get("d", 0))])
        for p in d.patches:
            if nodes[0] in p.nodes:
                sap_n_counts[p.sap] += 1
            if nodes[1] in p.nodes:
                sap_o_counts[p.sap] += 1
    sap_n = sap_n_counts.most_common(1)[0][0] if sap_n_counts else None
    sap_o = sap_o_counts.most_common(1)[0][0] if sap_o_counts else None
    rel_recs = [r for r in records if r["family"] == "relation"]

    def concept_for_sap(sap: str | None) -> int | None:
        if sap is None:
            return None
        counts = Counter(r["concept"] for r in rel_recs if r["sap"] == sap)
        return int(counts.most_common(1)[0][0]) if counts else None

    concept_n = concept_for_sap(sap_n)
    concept_o = concept_for_sap(sap_o)
    sens, support = pair_ablation_sensitivity(model, data, "relation", split)
    candidates = {k: v for k, v in sens.items() if support[k] >= pair_support_min}
    order = sorted(candidates, key=lambda k: -candidates[k])
    true_pair = None
    rank = None
    if concept_n is not None and concept_o is not None:
        best = None
        for pair in ((concept_n, concept_o), (concept_o, concept_n)):
            if pair in candidates:
                if best is None or candidates[pair] > candidates[best]:
                    best = pair
        true_pair = best
        if best is not None:
            rank = order.index(best) + 1
    top = [
        {"pair": [int(k1), int(k2)], "sensitivity": float(candidates[(k1, k2)]), "support": int(support[(k1, k2)])}
        for k1, k2 in order[:10]
    ]
    return {
        "sap_N": sap_n,
        "sap_O": sap_o,
        "concept_N": concept_n,
        "concept_O": concept_o,
        "true_pair": list(true_pair) if true_pair else None,
        "true_pair_sensitivity_rank": rank,
        "n_pairs_considered": len(candidates),
        "top_pairs": top,
    }


# ---------------------------------------------------------------------------
# gates / verdict
# ---------------------------------------------------------------------------


def decide_gates(m: Mapping) -> dict:
    gates: dict[str, dict] = {}

    preflight_ok = bool(m["preflight"]["passed"])
    gates["P_preflight"] = {"pass": preflight_ok, "note": "all five required checks passed"}

    g1 = {}
    for arm in ("A", "C"):
        acc = m["seed0"]["eval_test"][arm]["matched_pairs"]["relation"]["pair_balanced_accuracy"]
        g1[arm] = acc
    g1_pass = all(0.45 <= v <= 0.55 for v in g1.values())
    gates["G1_information_limit"] = {"pass": g1_pass, "A_pair_acc": g1["A"], "C_pair_acc": g1["C"], "band": [0.45, 0.55]}

    g2 = {}
    for arm in ("B", "D"):
        g2[arm] = m["seed0"]["eval_test"][arm]["matched_pairs"]["relation"]["pair_balanced_accuracy"]
    scr = {arm: m["seed0"]["scrambled_control"][arm]["matched_pairs"]["relation"]["pair_balanced_accuracy"] for arm in ("B", "D")}
    true_minus_scr = {arm: g2[arm] - scr[arm] for arm in ("B", "D")}
    g2_pass = all(v >= 0.80 for v in g2.values()) and all(v <= 0.60 for v in scr.values()) and all(v >= 0.15 for v in true_minus_scr.values())
    gates["G2_relation_used"] = {"pass": g2_pass, "true": g2, "scrambled": scr, "true_minus_scrambled": true_minus_scr}

    aud = m["seed0"]["audits"]["head_input_D"]
    gap = m["seed0"]["eval_test"]["D"]["hard_soft"]
    g3_pass = (
        bool(aud.get("exact_match"))
        and bool(aud.get("declared_names_match"))
        and gap["hard_balanced_accuracy"] >= 0.80
        and abs(gap["gap_soft_minus_hard"]) <= 0.05
    )
    gates["G3_hard_concept"] = {
        "pass": g3_pass,
        "audit_exact_match": bool(aud.get("exact_match")),
        "hard_minus_soft_gap": gap["gap_soft_minus_hard"],
        "hard_relation_pair_acc": g2["D"],
    }

    sem = m["seed0"]["concepts"]["D"]
    g4_pass = (
        sem["test_purity"]["matched_purity"] >= 0.80
        and sem["key_sap_purity"]["dominant_share"] >= 0.80
        and sem["cross_base"]["mean_base_agreement"] >= 0.80
        and (("cross_seed" not in sem) or (sem["cross_seed"].get("concept_assignment_agreement", 1.0) >= 0.80))
    )
    gates["G4_concept_semantics"] = {
        "pass": g4_pass,
        "test_purity": sem["test_purity"]["matched_purity"],
        "key_sap_purity": sem["key_sap_purity"]["dominant_share"],
        "cross_base": sem["cross_base"]["mean_base_agreement"],
        "cross_seed": sem.get("cross_seed", {}).get("concept_assignment_agreement"),
    }

    rec = m["seed0"]["explanations"]
    bind = rec["binding"]["D"] if "D" in rec["binding"] else next(iter(rec["binding"].values()))
    rel = rec["relation"]["D"] if "D" in rec["relation"] else next(iter(rec["relation"].values()))
    g5_pass = (
        (bind["top_key_concept_sensitivity_rank"] is not None and bind["top_key_concept_sensitivity_rank"] <= 3)
        or (bind["key_mass_spearman_vs_ablation_sensitivity"] >= 0.5)
    ) and (rel["true_pair_sensitivity_rank"] is not None and rel["true_pair_sensitivity_rank"] <= 5)
    gates["G5_explanation_recovery"] = {
        "pass": g5_pass,
        "binding_rank": bind["top_key_concept_sensitivity_rank"],
        "binding_spearman": bind["key_mass_spearman_vs_ablation_sensitivity"],
        "relation_pair_rank": rel["true_pair_sensitivity_rank"],
        "relation_top_pairs": rel["top_pairs"],
    }

    comp_pass = all(m["seed0"]["eval_test"][arm]["matched_pairs"]["composition"]["pair_balanced_accuracy"] >= 0.65 for arm in ("B", "D"))
    gates["G6_composition_generalization"] = {
        "pass_exploratory": comp_pass,
        "B": m["seed0"]["eval_test"]["B"]["matched_pairs"]["composition"]["pair_balanced_accuracy"],
        "D": m["seed0"]["eval_test"]["D"]["matched_pairs"]["composition"]["pair_balanced_accuracy"],
    }

    # soft leakage flag
    leak_flags = {}
    for arm in ("C", "D"):
        hs = m["seed0"]["eval_test"][arm]["hard_soft"]
        leak_flags[arm] = bool(
            hs["hard_balanced_accuracy"] < 0.70
            and hs["soft_balanced_accuracy"] >= 0.85
            and hs["gap_soft_minus_hard"] > 0.10
        )
    gates["soft_concept_leakage_flag"] = {"pass": not any(leak_flags.values()), "flags": leak_flags}

    verdict = "INCONCLUSIVE"
    reason = ""
    if not preflight_ok:
        verdict, reason = "INCONCLUSIVE", "preflight failed"
    elif any(leak_flags.values()):
        verdict, reason = "SOFT_CONCEPT_LEAKAGE", "hard concept fails while soft concepts succeed"
    elif not gates["G2_relation_used"]["pass"]:
        verdict, reason = "RELATION_NOT_USED", "true relations do not beat the label-blind scramble"
    elif not gates["G1_information_limit"]["pass"]:
        verdict, reason = "INCONCLUSIVE", "bag-only arms exceed chance on identical-input pairs"
    elif gates["G2_relation_used"]["pass"] and gates["G3_hard_concept"]["pass"] and gates["G4_concept_semantics"]["pass"] and gates["G5_explanation_recovery"]["pass"]:
        verdict, reason = "GROUNDING_AND_RELATIONS_SUPPORTED", "prediction, discrete concepts and explanation recovery all pass"
    elif gates["G2_relation_used"]["pass"] and gates["G3_hard_concept"]["pass"]:
        verdict, reason = "PREDICTION_ONLY_NO_GROUNDING", "prediction works but concept semantics/recovery fail"
    else:
        verdict, reason = "INCONCLUSIVE", "mixed evidence"
    gates["verdict"] = {"label": verdict, "reason": reason}
    return gates
