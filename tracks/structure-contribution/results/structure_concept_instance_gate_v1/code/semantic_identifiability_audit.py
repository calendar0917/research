"""Read-only semantic-identifiability audit for structure_concept_instance_gate_v1.

No training, no writes to frozen artifacts (REPORT/PROTOCOL/metrics are only
read).  Reuses the frozen concept records, the deterministic generator and the
frozen prediction records.

Output: analysis/semantic_identifiability_audit.json
"""

from __future__ import annotations

import itertools
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analysis as A  # noqa: E402
import synthetic as S  # noqa: E402

RESULTS = HERE.parent
ANALYSIS = RESULTS / "analysis"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def argmax_counter(counter: Counter, default):
    if not counter:
        return default
    return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]


def structural_index(g: S.Graph):
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from(g.edges)
    triangle = {}
    for w in G.nodes:
        nb = list(G.neighbors(w))
        triangle[w] = any(G.has_edge(a, b) for a, b in itertools.combinations(nb, 2))
    on_cycle = {w: False for w in G.nodes}
    for comp in nx.biconnected_components(G):
        if len(comp) >= 3:
            for w in comp:
                on_cycle[w] = True
    return G, triangle, on_cycle


def triangle_partners(G: nx.Graph, tri: dict[int, bool], w: int) -> tuple[int, int] | None:
    if not tri[w]:
        return None
    nb = list(G.neighbors(w))
    for a, b in itertools.combinations(nb, 2):
        if G.has_edge(a, b):
            return (a, b)
    return None


def cycle_visible_in_patch(G: nx.Graph, w: int, patch_nodes: set[int]) -> bool:
    nb = [x for x in G.neighbors(w) if x in patch_nodes]
    if len(nb) < 2:
        return False
    H = G.subgraph(patch_nodes - {w})
    return any(nx.has_path(H, a, b) for a, b in itertools.combinations(nb, 2))


def atom_role(g, G, triangle, on_cycle, patch) -> dict:
    anchors = set(patch.anchors)
    patch_nodes = set(patch.nodes)
    out = {}
    for w in patch.nodes:
        if g.attrs[w] == S.ATTR_C:
            continue
        el = S.ATTR_NAMES[g.attrs[w]]
        if triangle[w]:
            ring = "triangle"
        elif on_cycle[w]:
            ring = "ring"
        else:
            ring = "chain"
        partners = triangle_partners(G, triangle, w)
        tri_visible = bool(partners) and all(p in patch_nodes for p in partners)
        if tri_visible:
            closure = "triangle"
        elif cycle_visible_in_patch(G, w, patch_nodes):
            closure = "local_ring"
        else:
            closure = "no_local_closure"
        if w in anchors:
            pos = "anchor"
        elif all(G.has_edge(w, a) for a in anchors):
            pos = "common"
        else:
            pos = "neighbor"
        nbr_el = "".join(sorted(S.ATTR_NAMES[g.attrs[x]] for x in G.neighbors(w) if x != w))
        out[w] = {
            "element": el,
            "global_ring": ring,
            "local_closure": closure,
            "position": pos,
            "nbr_elements": nbr_el or "none",
            "triangle_visible": tri_visible,
        }
    return out


def patch_roles(g, G, triangle, on_cycle, patch) -> dict:
    atoms = atom_role(g, G, triangle, on_cycle, patch)
    atom_list = list(atoms.values())

    def multiset(field: str) -> tuple:
        vals = [a[field] for a in atom_list]
        return tuple(sorted(vals)) if vals else ("C_only",)

    out = {
        "role_element": multiset("element"),
        "role_global": multiset("global_ring"),
        "role_local": multiset("local_closure"),
        "n_special_atoms": len(atom_list),
    }
    if len(atom_list) == 1:
        a = atom_list[0]
        out["single_atom_element"] = a["element"]
        out["single_atom_global"] = f"{a['element']}|{a['global_ring']}|{a['position']}"
        out["single_atom_local"] = f"{a['element']}|{a['local_closure']}|{a['position']}"
        out["single_atom_rich_local"] = f"{a['element']}|{a['local_closure']}|{a['position']}|{a['nbr_elements']}"
        out["single_ring_triangle"] = "triangle" if a["global_ring"] == "triangle" else "not_triangle"
        if a["triangle_visible"]:
            out["single_ring_triangle_visible"] = "triangle_visible"
        elif a["global_ring"] == "triangle":
            out["single_ring_triangle_visible"] = "triangle_not_visible"
        else:
            out["single_ring_triangle_visible"] = "not_triangle"
    else:
        for k in (
            "single_atom_element",
            "single_atom_global",
            "single_atom_local",
            "single_atom_rich_local",
            "single_ring_triangle",
            "single_ring_triangle_visible",
        ):
            out[k] = None
    return out


def load_records(arm: str, seed: int) -> list[dict]:
    path = ANALYSIS / f"concepts_{arm}_seed{seed}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def attach_graph_roles(records: list[dict], graphs_by_id: dict[str, S.Graph], patches_by_key: dict) -> None:
    for r in records:
        g = graphs_by_id[r["graph_id"]]
        key = (r["graph_id"], r["patch_index"])
        patch = patches_by_key[key]
        assert patch.sap == r["sap"], (key, patch.sap, r["sap"])
        assert tuple(patch.nodes) == tuple(r["nodes"]), key
        G, tri, cyc = structural_index(g)
        r["feature"] = tuple(int(round(x)) for x in patch.feature)
        r.update(patch_roles(g, G, tri, cyc, patch))


def learn_role_map(records: list[dict], role_key: str, split: str = "train"):
    per_concept: dict[int, Counter] = defaultdict(Counter)
    pool: Counter = Counter()
    for r in records:
        if r["split"] != split or r.get(role_key) is None:
            continue
        per_concept[r["concept"]][r[role_key]] += 1
        pool[r[role_key]] += 1
    global_major = argmax_counter(pool, None)
    mapping = {c: argmax_counter(cnt, global_major) for c, cnt in per_concept.items()}
    return mapping, global_major, per_concept


def evaluate_role_map(records: list[dict], role_key: str, split: str, mapping: dict, global_major):
    sub = [r for r in records if r["split"] == split and r.get(role_key) is not None]
    n = len(sub)
    if n == 0:
        return {"n": 0}
    per_true: dict = defaultdict(lambda: [0, 0])
    unseen = 0
    for r in sub:
        if r["concept"] not in mapping:
            unseen += 1
        pred = mapping.get(r["concept"], global_major)
        per_true[r[role_key]][0] += 1
        if pred == r[role_key]:
            per_true[r[role_key]][1] += 1
    correct = sum(v[1] for v in per_true.values())
    recalls = {str(role): v[1] / v[0] for role, v in sorted(per_true.items(), key=lambda kv: str(kv[0]))}
    base_correct = sum(1 for r in sub if global_major == r[role_key])
    return {
        "n": n,
        "accuracy": correct / n,
        "macro_recall": float(np.mean(list(recalls.values()))),
        "n_roles": len(per_true),
        "per_role_recall": recalls,
        "per_role_support": {str(role): v[0] for role, v in sorted(per_true.items(), key=lambda kv: str(kv[0]))},
        "global_majority_role": str(global_major),
        "global_majority_baseline_accuracy": base_correct / n,
        "concept_unseen_in_train": unseen,
    }


def per_group_accuracy(records: list[dict], role_key: str, mapping: dict, global_major, split: str, group_key: str):
    groups: dict = defaultdict(lambda: [0, 0])
    for r in records:
        if r["split"] != split or r.get(role_key) is None:
            continue
        groups[r[group_key]][0] += 1
        if mapping.get(r["concept"], global_major) == r[role_key]:
            groups[r[group_key]][1] += 1
    return {str(k): {"n": v[0], "accuracy": v[1] / v[0]} for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def feature_lookup_ceiling(records: list[dict], role_key: str, split: str = "test"):
    per_feature: dict[tuple, Counter] = defaultdict(Counter)
    for r in records:
        if r["split"] == "train" and r.get(role_key) is not None:
            per_feature[r["feature"]][r[role_key]] += 1
    if not per_feature:
        return {}
    lookup = {f: argmax_counter(c, None) for f, c in per_feature.items()}
    sub = [r for r in records if r["split"] == split and r.get(role_key) is not None]
    correct = sum(1 for r in sub if lookup.get(r["feature"], None) == r[role_key])
    ambiguous = sum(1 for c in per_feature.values() if len(c) > 1)
    patches_ambiguous = sum(1 for r in sub if len(per_feature.get(r["feature"], Counter())) > 1)
    return {
        "n_test": len(sub),
        "lookup_accuracy": correct / len(sub) if sub else float("nan"),
        "n_train_features": len(per_feature),
        "n_ambiguous_features": ambiguous,
        "test_patches_with_ambiguous_feature": patches_ambiguous,
        "test_fraction_ambiguous_feature": patches_ambiguous / len(sub) if sub else float("nan"),
    }


# ---------------------------------------------------------------------------
# 1. metric audit
# ---------------------------------------------------------------------------


def collect_key_saps(graphs: list[S.Graph], data) -> tuple[set[str], int]:
    key_saps: set[str] = set()
    n_by_position = 0
    for g in graphs:
        if g.family != "binding":
            continue
        nodes = [g.meta.get("p_in"), g.meta.get("p_out")]
        for p in data[g.graph_id].patches:
            if any((n is not None and n in set(p.nodes)) for n in nodes):
                key_saps.add(p.sap)
                n_by_position += 1
    return key_saps, n_by_position


def _per_concept_majority_purity(records: list[dict], split: str) -> float:
    per_concept: dict[int, Counter] = defaultdict(Counter)
    for r in records:
        if r["split"] != split:
            continue
        per_concept[r["concept"]][r["sap"]] += 1
    n = sum(sum(c.values()) for c in per_concept.values())
    return sum(c.most_common(1)[0][1] for c in per_concept.values()) / n if n else float("nan")


def _top16_sap_faithful_purity(records: list[dict]) -> float:
    train_support = Counter(r["sap"] for r in records if r["split"] == "train")
    top = [s for s, _ in train_support.most_common(16)]
    ids = {s: i for i, s in enumerate(top)}
    modified = [dict(r, concept=ids.get(r["sap"], 16)) for r in records]
    return A.contingency_stats(modified, "test")["matched_purity"]


def metric_audit(graphs, data, all_records: dict) -> dict:
    key_saps, n_by_position = collect_key_saps(graphs, data)
    out: dict = {
        "key_sap_purity_definition": (
            "sub = patches whose SAP string is in the key set (any family, any split); "
            "dominant_share = share of sub assigned to the single most frequent concept. "
            "There is no SAP-to-concept matching and no recall term."
        ),
        "key_sap_set": {
            "n_saps": len(key_saps),
            "n_decisive_patches_by_position": n_by_position,
            "n_patches_matching_key_sap_anywhere": sum(
                1 for recs in all_records.values() for r in recs if r["sap"] in key_saps
            ),
        },
        "demos": {},
        "actual": {},
    }
    for arm, seed in sorted(all_records):
        recs = all_records[(arm, seed)]
        out["actual"][f"{arm}_seed{seed}"] = A.key_sap_purity(recs, key_saps)
        if arm != "D":
            continue
        sap_ids = {s: i for i, s in enumerate(sorted({r["sap"] for r in recs}))}
        exact = [dict(r, concept=sap_ids[r["sap"]]) for r in recs]
        const = [dict(r, concept=0) for r in recs]
        elem_ids: dict = {}
        for r in recs:
            key = tuple(sorted(set(r["role_element"])))
            elem_ids.setdefault(key, len(elem_ids))
        elem = [dict(r, concept=elem_ids[tuple(sorted(set(r["role_element"])))]) for r in recs]
        out["demos"][f"exact_sap_oracle_seed{seed}"] = A.key_sap_purity(exact, key_saps)
        out["demos"][f"constant_collapse_seed{seed}"] = A.key_sap_purity(const, key_saps)
        out["demos"][f"element_pattern_seed{seed}"] = A.key_sap_purity(elem, key_saps)

    hung: dict = {
        "definition": (
            "contingency_stats matched_purity = linear_sum_assignment over the concept x SAP count "
            "matrix, i.e. each concept is matched to at most one exact SAP and each SAP to at most "
            "one concept; unmatched mass scores zero. Many-to-one role abstraction is therefore "
            "punished in proportion to how many distinct exact SAPs a role spans."
        ),
        "per_seed": {},
    }
    for seed in (0, 1):
        recs = all_records[("D", seed)]
        tests = [r for r in recs if r["split"] == "test"]
        roles = sorted({tuple(r["role_global"]) for r in recs})
        role_id = {r_: i for i, r_ in enumerate(roles)}
        role_oracle = [dict(r, concept=role_id[tuple(r["role_global"])]) for r in recs]
        loc_roles = sorted({tuple(r["role_local"]) for r in recs})
        loc_id = {r_: i for i, r_ in enumerate(loc_roles)}
        local_oracle = [dict(r, concept=loc_id[tuple(r["role_local"])]) for r in recs]
        per_concept: dict[int, Counter] = defaultdict(Counter)
        for r in tests:
            per_concept[r["concept"]][r["sap"]] += 1
        n = len(tests)
        majority_purity = sum(c.most_common(1)[0][1] for c in per_concept.values()) / n
        sap_in_concept = {str(k): len(c) for k, c in sorted(per_concept.items())}
        hung["per_seed"][str(seed)] = {
            "actual_hungarian_purity_test": A.contingency_stats(recs, "test")["matched_purity"],
            "actual_n_concepts_used": A.contingency_stats(recs, "test")["n_concepts_used"],
            "actual_n_saps": A.contingency_stats(recs, "test")["n_saps"],
            "per_concept_majority_sap_purity_test": majority_purity,
            "saps_per_concept_test": sap_in_concept,
            "mean_saps_per_concept": float(np.mean(list(sap_in_concept.values()))),
            "oracle_global_role_hungarian_purity_test": A.contingency_stats(role_oracle, "test")["matched_purity"],
            "oracle_global_role_per_concept_majority_purity_test": _per_concept_majority_purity(role_oracle, "test"),
            "oracle_global_role_n_roles": len(roles),
            "oracle_local_role_hungarian_purity_test": A.contingency_stats(local_oracle, "test")["matched_purity"],
            "oracle_local_role_per_concept_majority_purity_test": _per_concept_majority_purity(local_oracle, "test"),
            "oracle_local_role_n_roles": len(loc_roles),
            "oracle_exact_sap_hungarian_purity_test": 1.0,
            "oracle_constant_hungarian_purity_test": A.contingency_stats([dict(r, concept=0) for r in recs], "test")["matched_purity"],
            "top16_train_sap_faithful_hungarian_purity_test": _top16_sap_faithful_purity(recs),
        }
    out["hungarian"] = hung
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    graphs = S.all_graphs()
    graphs_by_id = {g.graph_id: g for g in graphs}
    import pipeline as P

    data, manifest = P.prepare_data()
    patches_by_key = {(gid, j): p for gid, d in data.items() for j, p in enumerate(d.patches)}

    all_records: dict[tuple[str, int], list[dict]] = {}
    for arm in ("C", "D"):
        for seed in (0, 1):
            recs = load_records(arm, seed)
            attach_graph_roles(recs, graphs_by_id, patches_by_key)
            all_records[(arm, seed)] = recs

    metrics = {seed: json.loads((ANALYSIS / f"metrics_seed{seed}.json").read_text()) for seed in (0, 1)}

    out: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "inputs": {
            "concept_records": [f"analysis/concepts_{a}_seed{s}.jsonl" for a in ("C", "D") for s in (0, 1)],
            "metrics": ["analysis/metrics_seed0.json", "analysis/metrics_seed1.json"],
            "predictions": ["predictions/predictions_seed0.jsonl", "predictions/predictions_seed1.jsonl"],
            "graphs": "code/synthetic.py#all_graphs (deterministic regeneration, no training)",
        },
    }

    out["metric_audit"] = metric_audit(graphs, data, all_records)

    # ---- feature identifiability -----------------------------------------
    recs0 = all_records[("D", 0)]
    feats: dict = defaultdict(lambda: {"saps": set(), "roles": set(), "patches": 0})
    for r in recs0:
        f = r["feature"]
        feats[f]["saps"].add(r["sap"])
        feats[f]["roles"].add(tuple(r["role_global"]))
        feats[f]["patches"] += 1
    n_patches = sum(v["patches"] for v in feats.values())
    collided = {f: v for f, v in feats.items() if len(v["saps"]) > 1}
    role_collided = {f: v for f, v in feats.items() if len(v["roles"]) > 1}
    collision_examples = []
    for f, v in sorted(role_collided.items(), key=lambda kv: -len(kv[1]["saps"]))[:5]:
        saps = sorted(v["saps"])
        descs = [
            {"sap": sap[:8], "desc": next(r["sap_desc"] for r in recs0 if r["sap"] == sap)[:100]}
            for sap in saps[:3]
        ]
        collision_examples.append(
            {
                "n_saps": len(v["saps"]),
                "n_roles": len(v["roles"]),
                "roles": ["|".join(x) for x in sorted(v["roles"])][:4],
                "saps": descs,
            }
        )
    out["feature_identifiability"] = {
        "n_patch_instances": n_patches,
        "n_distinct_features": len(feats),
        "n_distinct_saps": len({r["sap"] for r in recs0}),
        "feature_groups_with_multiple_saps": len(collided),
        "patch_instances_in_multi_sap_feature_groups": sum(v["patches"] for v in collided.values()),
        "max_saps_per_feature": max((len(v["saps"]) for v in feats.values()), default=0),
        "feature_groups_merging_distinct_roles": len(role_collided),
        "patch_instances_in_role_merging_groups": sum(v["patches"] for v in role_collided.values()),
        "examples": collision_examples,
    }

    # ---- determinism checks ----------------------------------------------
    determinism: dict = {"per_arm_seed": {}, "interpretation": None}
    for (arm, seed), recs in sorted(all_records.items()):
        by_sap: dict[str, set[int]] = defaultdict(set)
        for r in recs:
            by_sap[r["sap"]].add(r["concept"])
        bad = {s: sorted(v) for s, v in by_sap.items() if len(v) > 1}
        determinism["per_arm_seed"][f"{arm}_seed{seed}"] = {
            "n_saps": len(by_sap),
            "n_saps_with_nonunique_concept": len(bad),
            "examples": dict(list(bad.items())[:3]),
        }
    determinism["interpretation"] = (
        "Same exact SAP -> same 35-d feature -> same encoder output -> same hard argmax, so every "
        "concept is a deterministic coarsening of SAP. Consequences: cross_base_consistency=1.0 is "
        "guaranteed up to tie-breaking, and I(concept;label|SAP)=0 is guaranteed by construction; "
        "neither is independent evidence of cross-base semantic generalisation."
    )
    out["determinism_checks"] = determinism

    # ---- role semantics ---------------------------------------------------
    role_keys = [
        "role_element",
        "single_atom_element",
        "single_atom_global",
        "single_atom_local",
        "single_atom_rich_local",
        "single_ring_triangle",
        "single_ring_triangle_visible",
    ]
    role_out: dict = {"per_arm_seed": {}, "single_atom_coverage": {}}
    for (arm, seed), recs in sorted(all_records.items()):
        entry: dict = {}
        for rk in role_keys:
            mapping, global_major, per_concept = learn_role_map(recs, rk, "train")
            entry[rk] = {
                "mapping_from_train": {str(k): str(v) for k, v in sorted(mapping.items())},
                "n_concepts_with_train_support": len(per_concept),
                "dev": evaluate_role_map(recs, rk, "dev", mapping, global_major),
                "test": evaluate_role_map(recs, rk, "test", mapping, global_major),
                "test_feature_lookup_ceiling": feature_lookup_ceiling(recs, rk, "test"),
            }
            if rk == "single_atom_local":
                entry[rk]["test_by_family"] = per_group_accuracy(recs, rk, mapping, global_major, "test", "family")
                entry[rk]["test_by_base"] = per_group_accuracy(recs, rk, mapping, global_major, "test", "base_id")
            if rk == "single_ring_triangle_visible":
                entry[rk]["test_by_family"] = per_group_accuracy(recs, rk, mapping, global_major, "test", "family")
        role_out["per_arm_seed"][f"{arm}_seed{seed}"] = entry
        n_total = len(recs)
        n_single = sum(1 for r in recs if r["n_special_atoms"] == 1)
        role_out["single_atom_coverage"][f"{arm}_seed{seed}"] = {
            "n_patches": n_total,
            "n_single_special_atom": n_single,
            "single_fraction": n_single / n_total,
            "n_special_atom_distribution": dict(sorted(Counter(r["n_special_atoms"] for r in recs).items())),
        }
    role_out["taxonomy_notes"] = (
        "role_element = multiset of special elements in the patch (coarse statistic). "
        "single_atom_* = only patches with exactly one special atom; global atom role includes "
        "'ring' (not observable from a radius-1 patch when the ring is > 3), the local version "
        "replaces ring/chain by the locally observable closure flag, the rich version adds the "
        "neighbour-element multiset. single_ring_triangle_visible distinguishes patches where the "
        "triangle is inside the patch from atoms globally in a triangle whose closure is not visible."
    )
    out["role_semantics"] = role_out

    # ---- extra: purity, coverage, confusion, information -----------------
    from sklearn.metrics import normalized_mutual_info_score, mutual_info_score

    extra: dict = {"per_arm_seed": {}}
    for (arm, seed), recs in sorted(all_records.items()):
        entry: dict = {}
        for rk in ("single_atom_element", "single_atom_local", "single_ring_triangle_visible"):
            mapping, global_major, per_concept = learn_role_map(recs, rk, "train")
            sub = [r for r in recs if r["split"] == "test" and r.get(rk) is not None]
            covered = [r for r in sub if r["concept"] in mapping]
            entry[rk] = {
                "test_coverage_concept_seen_in_train": len(covered) / len(sub) if sub else float("nan"),
                "test_accuracy_covered_subset": (
                    sum(1 for r in covered if mapping[r["concept"]] == r[rk]) / len(covered) if covered else float("nan")
                ),
                "mi_concept_role_test_nats": float(
                    mutual_info_score([str(r[rk]) for r in sub], [r["concept"] for r in sub])
                ),
                "mi_feature_role_test_nats": float(
                    mutual_info_score([str(r[rk]) for r in sub], [str(r["feature"]) for r in sub])
                ),
                "nmi_concept_role_test": float(
                    normalized_mutual_info_score([str(r[rk]) for r in sub], [r["concept"] for r in sub])
                ),
                "nmi_feature_role_test": float(
                    normalized_mutual_info_score([str(r[rk]) for r in sub], [str(r["feature"]) for r in sub])
                ),
                "concept_purity_test": {
                    str(c): {
                        "n": sum(1 for r in sub if r["concept"] == c),
                        "train_majority_role": str(mapping.get(c, global_major)),
                        "test_purity": (
                            sum(1 for r in sub if r["concept"] == c and r[rk] == mapping.get(c, global_major))
                            / max(sum(1 for r in sub if r["concept"] == c), 1)
                        ),
                        "test_role_counts": dict(
                            sorted(Counter(str(r[rk]) for r in sub if r["concept"] == c).items())
                        ),
                    }
                    for c in sorted({r["concept"] for r in sub})
                },
            }
            # role-level cross-base consistency (the honest counterpart of the
            # SAP-level metric, which is deterministic by construction)
            role_cells: dict = defaultdict(Counter)
            for r in recs:
                if r["split"] == "test" and r.get(rk) is not None:
                    role_cells[(str(r[rk]), r["base_id"])][r["concept"]] += 1
            by_role: dict = defaultdict(dict)
            for (role, base), cnt in role_cells.items():
                by_role[role][base] = argmax_counter(cnt, None)
            shares = []
            for role, per_base in by_role.items():
                if len(per_base) < 2:
                    continue
                global_dom = argmax_counter(Counter(per_base.values()), None)
                shares.append(sum(1 for c in per_base.values() if c == global_dom) / len(per_base))
            entry[rk]["role_cross_base_dominant_concept_agreement"] = float(np.mean(shares)) if shares else float("nan")
            entry[rk]["n_roles_cross_base"] = len(shares)
            # per-base accuracy with the per-base train-majority baseline
            by_base: dict = defaultdict(lambda: [0, 0, 0])
            for r in sub:
                by_base[r["base_id"]][0] += 1
                if mapping.get(r["concept"], global_major) == r[rk]:
                    by_base[r["base_id"]][1] += 1
                if global_major == r[rk]:
                    by_base[r["base_id"]][2] += 1
            entry[rk]["test_by_base_with_baseline"] = {
                str(b): {"n": v[0], "accuracy": v[1] / v[0], "majority_baseline": v[2] / v[0]}
                for b, v in sorted(by_base.items())
            }
        # element x concept confusion on single-atom patches (train and test)
        for split in ("train", "test"):
            conf: dict = defaultdict(Counter)
            for r in recs:
                if r["split"] == split and r["single_atom_element"] is not None:
                    conf[r["single_atom_element"]][r["concept"]] += 1
            entry[f"element_concept_confusion_{split}"] = {
                str(el): {str(c): n for c, n in sorted(cnt.items())} for el, cnt in sorted(conf.items())
            }
        # triangle-visible element x concept confusion on test single-atom patches
        conf: dict = defaultdict(Counter)
        for r in recs:
            if r["split"] == "test" and r["single_ring_triangle_visible"] == "triangle_visible":
                conf[r["single_atom_element"]][r["concept"]] += 1
        entry["triangle_visible_element_concept_confusion_test"] = {
            str(el): {str(c): n for c, n in sorted(cnt.items())} for el, cnt in sorted(conf.items())
        }
        extra["per_arm_seed"][f"{arm}_seed{seed}"] = entry
    extra["notes"] = (
        "coverage = share of test patches whose concept has train support for that role key; "
        "accuracy_covered = accuracy restricted to that subset. MI/nmi(concept;role) measures how "
        "much role information survives in the 16-prototype code on the test split; MI/nmi(feature;role) "
        "is the same for the raw 35-d input, i.e. the information available before the bottleneck "
        "(both on the same test sample, so MI(concept;role) <= MI(feature;role) by data processing). "
        "concept_purity_test reports, per concept, the share of its test patches matching its "
        "train-majority role (purity) next to the per-role recall above. "
        "role_cross_base_dominant_concept_agreement is the role-level analogue of the frozen "
        "SAP-level cross_base_consistency (which is deterministic by construction)."
    )
    out["role_purity_coverage_confusion"] = extra

    # ---- explanations recheck --------------------------------------------
    expl: dict = {}
    for seed in (0, 1):
        m = metrics[seed]
        entry: dict = {}
        for arm in ("C", "D"):
            rel = m["explanations"]["relation"][arm]
            bind = m["explanations"]["binding"][arm]
            top_pair = rel["top_pairs"][0]["pair"] if rel["top_pairs"] else None
            entry[arm] = {
                "relation_concept_N": rel["concept_N"],
                "relation_concept_O": rel["concept_O"],
                "N_and_O_same_concept": rel["concept_N"] == rel["concept_O"],
                "true_pair": rel["true_pair"],
                "true_pair_rank": rel["true_pair_sensitivity_rank"],
                "top_pair": top_pair,
                "top_pair_is_self_pair": bool(top_pair and top_pair[0] == top_pair[1]),
                "top_pair_sensitivity": rel["top_pairs"][0]["sensitivity"] if rel["top_pairs"] else None,
                "binding_top_key_concept": bind["top_key_concept"],
                "binding_rank": bind["top_key_concept_sensitivity_rank"],
            }
            topk = bind.get("top_key_concept")
            if topk is not None:
                sub = [r for r in all_records[(arm, seed)] if r["concept"] == topk and r["family"] == "binding"]
                dist = Counter(
                    f"{a['element']}:{a['global_ring']}" for r in sub for a in atom_role(
                        graphs_by_id[r["graph_id"]],
                        *structural_index(graphs_by_id[r["graph_id"]])[:3],
                        patches_by_key[(r["graph_id"], r["patch_index"])],
                    ).values()
                )
                entry[arm]["binding_top_concept_atom_roles"] = dict(dist.most_common())
        expl[str(seed)] = entry
    out["explanations_recheck"] = expl

    # ---- composition selection -------------------------------------------
    comp: dict = {"dev_composition_graphs": 0, "train_composition_graphs": 0, "test_composition_graphs": 0}
    for g in graphs:
        if g.family == "composition":
            comp[f"{g.split}_composition_graphs"] += 1
    comp["model_selection"] = (
        "train_arm selects the checkpoint by dev balanced accuracy over all 16 dev graphs, "
        "4 of which are composition; composition labels therefore influenced checkpoint selection "
        "even though no composition graph contributes training gradients."
    )
    per_arm: dict = {}
    for seed in (0, 1):
        rows = [
            json.loads(line)
            for line in (RESULTS / "predictions" / f"predictions_seed{seed}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for arm in ("C", "D"):
            sub = [r for r in rows if r["arm"] == arm and r["family"] == "composition"]
            for split in ("dev", "test"):
                s = [r for r in sub if r["split"] == split]
                if s:
                    acc = float(np.mean([(r["hard_prob"] >= 0.5) == (r["label"] > 0) for r in s]))
                    per_arm[f"{arm}_seed{seed}_{split}"] = {"n": len(s), "hard_accuracy": acc}
    comp["dev_vs_test_composition_accuracy"] = per_arm
    out["composition_selection"] = comp

    out_path = ANALYSIS / "semantic_identifiability_audit.json"
    out_path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
