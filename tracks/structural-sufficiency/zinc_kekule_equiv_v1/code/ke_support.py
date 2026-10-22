"""Train-only support analysis under verified Kekule re-representations.

Four vocabularies are built from official-train molecules only:

``orig``   the committed Kekule representation (old radius-2 content keys);
``rep``    canonical key = lexicographic minimum over the verified legal
           Kekule orbit of the *same atom occurrence* (representation quotient);
``arom``   ordinary aromaticisation: every verified aromatic bond is marked
           with one aromatic token (what a plain "aromatic bond" key would do);
``union``  union of all verified representation keys (joint statistics).

The old grouping (radius-2 content key relative to train, crossed with
``y > -3``) is reconstructed from the frozen reference CSV and never replaced.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import ke_chem as KC  # noqa: E402
import ke_graph as KG  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"
REFERENCE = (KG.SS / "zinc_restricted_adaptation_v1/analysis/reference_preds_valid.csv")

GROUP_NAMES = ("seen_ordinary", "seen_tail", "unseen_ordinary", "unseen_tail")
SCHEMES = ("orig", "rep", "arom", "union")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_valid_artifact(path: Path = ANALYSIS / "valid_reps.npz") -> dict:
    d = KG.load_reps_artifact(path)
    meta = json.loads(str(d["meta_json"][0]))
    return {"arrays": d, "meta": meta}


def valid_records_from_artifact(art: dict, ds) -> list[dict]:
    a = art["arrays"]
    kinds = {0: "exhaustive", 1: "no_alternative", 2: "truncated", 3: "unsupported"}
    out = []
    for i in range(len(ds)):
        ne = int(a["n_edges"][i])
        nr = int(a["n_reps"][i])
        edges = [(int(a["edge_u"][i, k]), int(a["edge_v"][i, k])) for k in range(ne)]
        orig = np.asarray(a["orig_order"][i, :ne], dtype=np.int64)
        reps = [np.asarray(a["rep_orders"][i, j, :ne], dtype=np.int64) for j in range(nr)]
        arom_mask = np.asarray(a["aromatic_mask"][i, :ne], dtype=bool)
        out.append({
            "row": i,
            "edges": edges,
            "orig_orders": orig,
            "reps": reps,
            "aromatic_edge_set": {edges[k] for k in range(ne) if arom_mask[k]},
            "status": kinds[int(a["status"][i])],
            "truncated": bool(a["truncated"][i]),
            "n_total": int(a["n_total"][i]),
            "n_inferred_h": int(a["n_inferred_h"][i]),
            "graph_sha256": str(a["graph_sha256"][i]),
        })
    return out


def load_train_records(ds, max_reps: int = 8) -> tuple[list[dict], list[dict]]:
    records = []
    audits = []
    for i in range(len(ds)):
        cats, edges, orig_orders, ei, ea = KG.graph_arrays(ds[i])
        res = KC.generate_representations(cats, edges, orig_orders, max_reps=max_reps)
        reps = list(res.reps) if res.reps else [orig_orders.copy()]
        rec = {
            "row": i,
            "cats": cats,
            "edges": edges,
            "orig_orders": orig_orders,
            "reps": reps,
            "aromatic_edge_set": set(res.aromatic_edges),
            "status": res.status,
            "truncated": bool(res.truncated),
            "n_total": int(res.n_total),
            "n_inferred_h": int(res.n_inferred_h),
            "h_audit": res.h_audit,
            "total_h": res.total_h,
        }
        records.append(rec)
        if res.h_audit:
            audits.append({"row": i, **res.h_audit})
    return records, audits


def load_reference() -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(REFERENCE, newline="", encoding="utf-8")))
    if len(rows) != 1000:
        raise RuntimeError(f"reference valid rows {len(rows)} != 1000")
    if not np.array_equal(np.asarray([int(r["row"]) for r in rows]), np.arange(1000)):
        raise RuntimeError("reference rows are not 0..999")
    y = np.asarray([float(r["y_raw"]) for r in rows])
    unseen = np.asarray([int(r["signature_unseen"]) for r in rows])
    ordinary = np.asarray([int(r["ordinary_y_gt_minus3"]) for r in rows])
    frac = np.asarray([float(r["r2_unseen_atom_frac"]) for r in rows])
    if not np.array_equal(unseen, (frac > 0).astype(np.int64)):
        raise RuntimeError("reference unseen flag disagrees with fraction")
    if not np.array_equal(ordinary, (y > -3.0).astype(np.int64)):
        raise RuntimeError("reference ordinary flag disagrees with y")
    group = np.empty(1000, dtype=object)
    group[(unseen == 0) & (y > -3)] = "seen_ordinary"
    group[(unseen == 0) & (y <= -3)] = "seen_tail"
    group[(unseen == 1) & (y > -3)] = "unseen_ordinary"
    group[(unseen == 1) & (y <= -3)] = "unseen_tail"
    return {"y": y, "unseen": unseen, "ordinary": ordinary,
            "old_unseen_frac": frac, "group": group}


def aromatic_orders(edges: Sequence[tuple[int, int]], orders: np.ndarray,
                    aromatic_edge_set: set[tuple[int, int]]) -> np.ndarray:
    out = np.asarray(orders).copy()
    for k, e in enumerate(edges):
        if e in aromatic_edge_set:
            out[k] = KC.AROMATIC_TOKEN
    return out


# ---------------------------------------------------------------------------
# vocabulary / per-atom support
# ---------------------------------------------------------------------------


def _key_has_arom(key: tuple) -> bool:
    c, nbrs = key
    return any(int(t) == KC.AROMATIC_TOKEN or _key_has_arom(u) for t, u in nbrs)


def build_vocab(train_records: Sequence[dict]) -> tuple[dict[str, set], dict[str, Any]]:
    V = {s: set() for s in SCHEMES}
    orig_min: dict[Any, set] = defaultdict(set)
    min_orig: dict[Any, set] = defaultdict(set)
    orig_arom: dict[Any, set] = defaultdict(set)
    arom_orig: dict[Any, set] = defaultdict(set)
    witness: dict[Any, set] = defaultdict(set)
    key_example: dict[Any, tuple[int, int]] = {}
    key_occurrences: dict[Any, list[tuple[int, int, Any]]] = defaultdict(list)
    arom_h: dict[Any, set[int]] = defaultdict(set)
    n_occ = 0
    n_multi_orbit = 0
    truncated_occ = 0
    for rec in train_records:
        cats, edges = rec["cats"], rec["edges"]
        k0 = KG.radius_keys(cats, edges, rec["orig_orders"])
        kr = [KG.radius_keys(cats, edges, r) for r in rec["reps"]]
        ka = KG.radius_keys(cats, edges,
                            aromatic_orders(edges, rec["orig_orders"],
                                            rec["aromatic_edge_set"]))
        for k in k0:
            V["orig"].add(k)
        for ks in zip(*kr):
            V["rep"].add(min(ks))
        for ks in kr:
            for k in ks:
                V["union"].add(k)
        for k in ka:
            V["arom"].add(k)
        for i in range(len(cats)):
            n_occ += 1
            o = k0[i]
            m = min(ks[i] for ks in kr)
            a = ka[i]
            orig_min[o].add(m)
            min_orig[m].add(o)
            orig_arom[o].add(a)
            arom_orig[a].add(o)
            key_example.setdefault(o, (rec["row"], i))
            if len(key_occurrences[o]) < 8:
                key_occurrences[o].append((rec["row"], i, a))
            if i < len(rec.get("total_h", [])):
                arom_h[a].add(int(rec["total_h"][i]))
            if rec["truncated"]:
                truncated_occ += 1
            orbit = {ks[i] for ks in kr}
            if len(orbit) > 1:
                n_multi_orbit += 1
                ol = list(orbit)
                for x in range(len(ol)):
                    for y in range(x + 1, len(ol)):
                        witness[ol[x]].add(ol[y])
                        witness[ol[y]].add(ol[x])
    stats = {
        "n_occurrences": int(n_occ),
        "n_multi_orbit_occurrences": int(n_multi_orbit),
        "n_truncated_occurrences": int(truncated_occ),
        "n_orig_types": len(V["orig"]),
        "n_rep_types": len(V["rep"]),
        "n_arom_types": len(V["arom"]),
        "n_union_types": len(V["union"]),
        "orig_keys_with_multiple_rep_images": int(sum(1 for v in orig_min.values() if len(v) > 1)),
        "orig_keys_with_multiple_arom_images": int(sum(1 for v in orig_arom.values() if len(v) > 1)),
        "rep_merge_classes": int(sum(1 for v in min_orig.values() if len(v) > 1)),
        "arom_merge_classes": int(sum(1 for v in arom_orig.values() if len(v) > 1)),
        "rep_merged_extra_keys": int(sum(len(v) - 1 for v in min_orig.values() if len(v) > 1)),
        "arom_merged_extra_keys": int(sum(len(v) - 1 for v in arom_orig.values() if len(v) > 1)),
        "rep_keys_not_original": int(len(V["rep"] - V["orig"])),
        "arom_keys_not_original": int(len(V["arom"] - V["orig"])),
        "union_keys_not_original": int(len(V["union"] - V["orig"])),
        "arom_classes_with_multiple_total_h": int(sum(
            1 for k, v in arom_h.items() if len(v) > 1 and _key_has_arom(k))),
    }
    stats.update(_merge_component_stats(min_orig, witness, "rep"))
    stats.update(_merge_component_stats(arom_orig, witness, "arom"))
    context = {
        "orig_min": orig_min, "min_orig": min_orig,
        "orig_arom": orig_arom, "arom_orig": arom_orig,
        "witness": witness, "key_example": key_example,
        "key_occurrences": key_occurrences,
        "arom_h": arom_h,
    }
    return V, {"stats": stats, "context": context}


def _components(nodes: Iterable[Any], witness: dict[Any, set]) -> list[list[Any]]:
    nodes = set(nodes)
    seen: set[Any] = set()
    out = []
    for s in nodes:
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in witness.get(u, ()):
                if v in nodes and v not in seen:
                    seen.add(v)
                    stack.append(v)
        out.append(comp)
    return out


def _merge_component_stats(classes: dict[Any, set], witness: dict[Any, set],
                           name: str) -> dict:
    merged = [v for v in classes.values() if len(v) > 1]
    multi = 0
    unwitnessed = 0
    max_comp = 0
    for v in merged:
        comps = _components(v, witness)
        if len(comps) > 1:
            multi += 1
            unwitnessed += len(v) - len(comps)
        max_comp = max(max_comp, len(comps))
    return {
        f"{name}_merged_classes_with_multiple_witness_components": int(multi),
        f"{name}_unwitnessed_merged_extra_keys": int(unwitnessed),
        f"{name}_max_witness_components_in_a_merged_class": int(max_comp),
    }


def valid_support(valid_records: Sequence[dict], ds, V: dict[str, set]) -> tuple[list[dict], list[dict]]:
    per_mol = []
    per_atom = []
    for rec in valid_records:
        row = rec["row"]
        cats, edges = KG.graph_arrays(ds[row])[0], rec["edges"]
        k0 = KG.radius_keys(cats, edges, rec["orig_orders"])
        kr = [KG.radius_keys(cats, edges, r) for r in rec["reps"]]
        krep = [min(ks) for ks in zip(*kr)]
        ka = KG.radius_keys(cats, edges,
                            aromatic_orders(edges, rec["orig_orders"],
                                            rec["aromatic_edge_set"]))
        s0 = np.asarray([k in V["orig"] for k in k0], dtype=bool)
        sr = np.asarray([k in V["rep"] for k in krep], dtype=bool)
        sa = np.asarray([k in V["arom"] for k in ka], dtype=bool)
        su = np.asarray([any(k in V["union"] for k in ks) for ks in zip(*kr)], dtype=bool)
        per_atom.append({"row": row, "s0": s0, "sr": sr, "sa": sa, "su": su})
        per_mol.append({
            "row": row,
            "n_atoms": len(cats),
            "n_aromatic_edges": len(rec["aromatic_edge_set"]),
            "status": rec["status"],
            "n_reps": len(rec["reps"]),
            "truncated": bool(rec["truncated"]),
            "n_total": int(rec["n_total"]),
            "n_inferred_h": int(rec["n_inferred_h"]),
            "unseen_orig": int((~s0).sum()),
            "unseen_rep": int((~sr).sum()),
            "unseen_arom": int((~sa).sum()),
            "unseen_union": int((~su).sum()),
            "full_orig": bool(s0.all()),
            "full_rep": bool(sr.all()),
            "full_arom": bool(sa.all()),
            "full_union": bool(su.all()),
            "rescued_union": int(((~s0) & su).sum()),
            "lost_rep": int((s0 & ~sr).sum()),
            "lost_arom": int((s0 & ~sa).sum()),
        })
    return per_mol, per_atom


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def _mean(x) -> float:
    return float(np.mean(x)) if len(x) else float("nan")


def _median(x) -> float:
    return float(np.median(x)) if len(x) else float("nan")


def aggregate(per_mol: Sequence[dict], ref: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    group = ref["group"]
    masks = {"all": np.ones(1000, dtype=bool)}
    for g in GROUP_NAMES:
        masks[g] = group == g
    for gname, mask in masks.items():
        rows = [r for r in per_mol if mask[r["row"]]]
        n = len(rows)
        atoms = sum(r["n_atoms"] for r in rows)
        block: dict[str, Any] = {
            "n_molecules": int(n),
            "n_atoms": int(atoms),
            "n_aromatic_molecules": int(sum(1 for r in rows if r["n_aromatic_edges"] > 0)),
            "n_no_alternative": int(sum(1 for r in rows if r["status"] == "no_alternative")),
            "n_truncated": int(sum(1 for r in rows if r["truncated"])),
            "n_unsupported": int(sum(1 for r in rows if r["status"] == "unsupported")),
            "n_rescued_atoms_union": int(sum(r["rescued_union"] for r in rows)),
            "n_lost_atoms_rep": int(sum(r["lost_rep"] for r in rows)),
            "n_lost_atoms_arom": int(sum(r["lost_arom"] for r in rows)),
            "schemes": {},
        }
        for s in SCHEMES:
            unseen_key = f"unseen_{s}"
            full_key = f"full_{s}"
            unseen_atoms = int(sum(r[unseen_key] for r in rows))
            fracs = np.asarray([r[unseen_key] / r["n_atoms"] for r in rows], dtype=float)
            block["schemes"][s] = {
                "unseen_atoms": unseen_atoms,
                "unseen_atom_fraction_pooled": float(unseen_atoms / atoms) if atoms else float("nan"),
                "unseen_atom_fraction_mean_per_molecule": _mean(fracs),
                "unseen_atom_fraction_median_per_molecule": _median(fracs),
                "fully_supported_molecules": int(sum(r[full_key] for r in rows)),
                "fully_supported_fraction": float(sum(r[full_key] for r in rows) / n) if n else float("nan"),
            }
        out[gname] = block
    return out


# ---------------------------------------------------------------------------
# examples
# ---------------------------------------------------------------------------


def _readable_key(key: tuple) -> str:
    def bt(x: int) -> str:
        return {1: "S", 2: "D", 3: "T", KC.AROMATIC_TOKEN: "A"}.get(int(x), f"b{x}")

    def sig(node) -> str:
        c, nbrs = node
        return f"{KC.atom_word(c)}(" + ",".join(f"{bt(t)}:{KC.atom_word(u)}" for t, u in nbrs) + ")"

    c, nbrs = key
    return f"{KC.atom_word(c)}(" + ",".join(f"{bt(t)}:{sig(u)}" for t, u in nbrs) + ")"


def _local_bonds(rec: dict, atom: int) -> list[dict]:
    out = []
    for k, (u, v) in enumerate(rec["edges"]):
        if atom not in (u, v):
            continue
        other = v if u == atom else u
        out.append({"other": int(other), "other_word": KC.atom_word(int(rec["cats"][other])),
                    "order": int(rec["orig_orders"][k]),
                    "aromatic": (u, v) in rec["aromatic_edge_set"]})
    return out


def _rep_local_bonds(rec: dict, atom: int, rep_index: int) -> list[dict]:
    out = []
    for k, (u, v) in enumerate(rec["edges"]):
        if atom not in (u, v):
            continue
        other = v if u == atom else u
        out.append({"other": int(other), "order": int(rec["reps"][rep_index][k]),
                    "aromatic": (u, v) in rec["aromatic_edge_set"]})
    return out


def build_examples(valid_records, ds, train_records, V, ctx, per_atom, ref,
                   max_examples: int = 6) -> dict[str, Any]:
    by_row = {r["row"]: r for r in valid_records}
    # 1. valid atoms in the 225 whose original key is unseen but a legal rep is seen
    rescue = []
    for rec, pa in zip(valid_records, per_atom):
        if ref["group"][rec["row"]] != "unseen_ordinary":
            continue
        for atom in range(len(pa["s0"])):
            if pa["s0"][atom] or not pa["su"][atom]:
                continue
            # find the first rep key that is supported
            cats, edges = KG.graph_arrays(ds[rec["row"]])[0], rec["edges"]
            k0 = KG.radius_keys(cats, edges, rec["orig_orders"])[atom]
            reps = [KG.radius_keys(cats, edges, r)[atom] for r in rec["reps"]]
            supported = [j for j, k in enumerate(reps) if k in V["union"]]
            rescue.append({
                "row": int(rec["row"]),
                "atom": int(atom),
                "atom_word": KC.atom_word(int(cats[atom])),
                "original_key": _readable_key(k0),
                "local_bonds_original": _local_bonds(rec, atom),
                "supported_rep_indices": supported,
                "rep_local_bonds": [_rep_local_bonds(rec, atom, j) for j in supported[:2]],
                "n_reps": len(rec["reps"]),
                "status": rec["status"],
            })
            if len(rescue) >= max_examples:
                break
        if len(rescue) >= max_examples:
            break

    # 2. representative witness merge (rep) and an unwitnessed aromatic merge
    mol_cache: dict[int, Any] = {}

    def key_occurrence(row_atom):
        row, atom = row_atom
        rec = train_records[row] if row < len(train_records) else None
        if rec is None:
            return None
        cats = rec["cats"]
        if rec.get("total_h"):
            total_h = int(rec["total_h"][int(atom)])
        else:
            if row not in mol_cache:
                mol_cache[row] = KC.build_mol_from_graph(cats, rec["edges"],
                                                         rec["orig_orders"])
            mol = mol_cache[row]
            total_h = int(mol.GetAtomWithIdx(int(atom)).GetTotalNumHs())
        dh = int(KC.dictionary_h(cats)[atom])
        return {
            "row": int(row), "atom": int(atom),
            "atom_word": KC.atom_word(int(cats[atom])),
            "dictionary_h": dh,
            "total_h": total_h,
            "inferred_h": bool(total_h != dh),
            "local_bonds": _local_bonds(rec, atom),
        }

    witness_examples = []
    for cls, origs in ctx["min_orig"].items():
        if len(origs) < 2:
            continue
        comps = _components(origs, ctx["witness"])
        if len(comps) != 1:
            continue
        occ = [key_occurrence(ctx["key_example"][o]) for o in comps[0][:3]]
        witness_examples.append({
            "canonical_key": _readable_key(cls),
            "original_keys": [_readable_key(o) for o in comps[0][:3]],
            "occurrences": [o for o in occ if o],
            "witness_components": 1,
        })
        if len(witness_examples) >= 2:
            break

    overmerge_examples = []
    overmerge_candidates = []
    for cls, origs in ctx["arom_orig"].items():
        if len(origs) < 2:
            continue
        comps = _components(origs, ctx["witness"])
        if len(comps) < 2:
            continue
        occs = []
        for comp in comps[:3]:
            for o in comp[:2]:
                for row, atom, a in ctx["key_occurrences"].get(o, [])[:1]:
                    occ = key_occurrence((row, atom))
                    if occ:
                        occ["original_key"] = _readable_key(o)
                        occ["aromatic_key"] = _readable_key(a)
                        occs.append(occ)
        overmerge_candidates.append({
            "aromatic_key": _readable_key(cls),
            "n_original_keys": len(origs),
            "n_witness_components": len(comps),
            "n_distinct_total_h": len(ctx["arom_h"].get(cls, set())),
            "total_h_values": sorted(int(x) for x in ctx["arom_h"].get(cls, set())),
            "occurrences": occs,
        })
    overmerge_candidates.sort(key=lambda x: (-x["n_distinct_total_h"],
                                              -x["n_witness_components"],
                                              -x["n_original_keys"]))
    overmerge_examples = overmerge_candidates[:3]

    # 3. context split: one original key with >1 aromatic image
    split_examples = []
    for o, occs in ctx["key_occurrences"].items():
        by_image: dict[Any, tuple[int, int]] = {}
        for row, atom, a in occs:
            by_image.setdefault(a, (row, atom))
        if len(by_image) < 2:
            continue
        rows = []
        for img, (row, atom) in list(by_image.items())[:3]:
            ex = key_occurrence((row, atom))
            if ex:
                rows.append({"aromatic_key": _readable_key(img), "occurrence": ex})
        split_examples.append({"original_key": _readable_key(o), "images": rows})
        if len(split_examples) >= 2:
            break

    return {
        "rescued_atoms_in_225": rescue,
        "witness_merge_examples": witness_examples,
        "unwitnessed_aromatic_merge_examples": overmerge_examples,
        "context_split_examples": split_examples,
    }


# ---------------------------------------------------------------------------
# H audit aggregation
# ---------------------------------------------------------------------------


def h_audit_summary(valid_records, valid_audits, train_records, train_audits) -> dict:
    def merge(audits):
        c = Counter()
        examples = []
        for a in audits:
            if a.get("explicit_h_no_solution"):
                c["explicit_h_no_solution"] += 1
            if a.get("explicit_h_original_missing"):
                c["explicit_h_original_missing"] += 1
            if a.get("explicit_h_h_changing_candidates"):
                c["explicit_h_h_changing_candidates"] += 1
            if (a.get("explicit_h_no_solution") or a.get("explicit_h_original_missing")
                    or a.get("explicit_h_h_changing_candidates")):
                if len(examples) < 6:
                    examples.append(a)
        return {"n_molecules_with_inferred_h": len(audits),
                "counts": dict(c), "examples": examples}

    return {
        "valid": merge(valid_audits),
        "train": merge(train_audits),
        "note": ("The committed atom dictionary does not encode neutral N/O/S H. "
                 "The audit enumerates with dictionary-only H; failures/candidates "
                 "there show why the frozen inferred-H state is required. Those "
                 "candidates are never used."),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-reps", type=int, default=8)
    args = p.parse_args(argv)

    art = load_valid_artifact()
    valid_ds = KG.load_pyg("valid")
    valid_records = valid_records_from_artifact(art, valid_ds)
    for rec in valid_records:
        rec["cats"] = KG.graph_arrays(valid_ds[rec["row"]])[0]
    train_ds = KG.load_pyg("train")
    train_records, train_audits = load_train_records(train_ds, args.max_reps)
    ref = load_reference()

    # verify the old 225 / 251 split and the artifact graph hashes
    if int((ref["group"] == "unseen_ordinary").sum()) != 225:
        raise RuntimeError("old unseen+ordinary count is not 225")
    if int((ref["unseen"] == 1).sum()) != 251:
        raise RuntimeError("old unseen count is not 251")
    for rec in valid_records:
        cats, edges, orig, ei, ea = KG.graph_arrays(valid_ds[rec["row"]])
        if KG.graph_sha256(cats, ei, ea) != rec["graph_sha256"]:
            raise RuntimeError(f"valid row {rec['row']}: graph hash mismatch")

    # H audit for valid records is recomputed only for inferred-H molecules
    valid_audits = []
    for rec in valid_records:
        if rec["n_inferred_h"] <= 0:
            continue
        cats, edges, orig, ei, ea = KG.graph_arrays(valid_ds[rec["row"]])
        ar_edges = sorted(rec["aromatic_edge_set"])
        orig_ar = tuple(int(rec["orig_orders"][rec["edges"].index(e)]) for e in ar_edges)
        aud = KC.explicit_h_only_audit(cats, edges, orig, ar_edges, orig_ar)
        if aud.get("status") == "ok":
            valid_audits.append({"row": rec["row"], **aud})

    V, vocab_ctx = build_vocab(train_records)
    per_mol, per_atom = valid_support(valid_records, valid_ds, V)

    # old unseen fraction reproduction check (per-molecule mean over atoms)
    old_repro = np.asarray([
        (r["unseen_orig"] / r["n_atoms"]) for r in per_mol
    ])
    max_old_diff = float(np.max(np.abs(old_repro - ref["old_unseen_frac"])))

    agg = aggregate(per_mol, ref)
    examples = build_examples(valid_records, valid_ds, train_records, V,
                              vocab_ctx["context"], per_atom, ref)
    h_audit = h_audit_summary(valid_records, valid_audits, train_records, train_audits)

    # per-molecule CSV
    out_csv = ANALYSIS / "support_per_molecule.csv"
    fields = ["row", "old_group", "y", "n_atoms", "n_aromatic_edges", "status",
              "n_reps", "truncated", "n_total", "n_inferred_h",
              "unseen_orig", "unseen_rep", "unseen_arom", "unseen_union",
              "frac_unseen_orig", "frac_unseen_rep", "frac_unseen_arom", "frac_unseen_union",
              "full_orig", "full_rep", "full_arom", "full_union",
              "rescued_union", "lost_rep", "lost_arom"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in per_mol:
            row = {
                **r,
                "old_group": str(ref["group"][r["row"]]),
                "y": float(ref["y"][r["row"]]),
                "frac_unseen_orig": r["unseen_orig"] / r["n_atoms"],
                "frac_unseen_rep": r["unseen_rep"] / r["n_atoms"],
                "frac_unseen_arom": r["unseen_arom"] / r["n_atoms"],
                "frac_unseen_union": r["unseen_union"] / r["n_atoms"],
            }
            w.writerow({k: row[k] for k in fields})

    summary = {
        "protocol": "zinc-kekule-equiv-v1",
        "split": "official PyG ZINC subset valid, n=1000; old groups fixed",
        "official_test_loaded": False,
        "rdkit_version": art["meta"].get("rdkit_version", KC.rdkit_version()),
        "old_group_check": {
            "n_unseen": int((ref["unseen"] == 1).sum()),
            "n_unseen_ordinary": int((ref["group"] == "unseen_ordinary").sum()),
            "max_abs_diff_old_unseen_fraction": max_old_diff,
            "matches_reference": bool(max_old_diff < 1e-9),
        },
        "representation_status": {
            "valid": dict(Counter(r["status"] for r in valid_records)),
            "train": dict(Counter(r["status"] for r in train_records)),
            "valid_truncated": int(sum(1 for r in valid_records if r["truncated"])),
            "train_truncated": int(sum(1 for r in train_records if r["truncated"])),
            "valid_unsupported": int(sum(1 for r in valid_records if r["status"] == "unsupported")),
            "train_unsupported": int(sum(1 for r in train_records if r["status"] == "unsupported")),
        },
        "vocabulary": vocab_ctx["stats"],
        "groups": agg,
        "h_audit": h_audit,
        "examples": examples,
        "definitions": {
            "orig": "committed Kekule representation; old radius-2 content keys",
            "rep": "minimum key over the verified legal Kekule orbit of the same atom occurrence",
            "arom": "all verified aromatic bonds replaced by one aromatic token (ordinary aromaticisation)",
            "union": "union of all keys over verified legal representations (joint statistics)",
            "full_support": "every atom occurrence in the molecule has a key in the train vocabulary",
            "unseen_atom_fraction": "pooled unseen atom occurrences divided by all atom occurrences (and per-molecule mean separately)",
        },
    }
    (ANALYSIS / "support_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ANALYSIS / "support_examples.json").write_text(
        json.dumps(examples, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tables(summary)
    print(json.dumps({k: summary[k] for k in ("old_group_check", "representation_status",
                                             "vocabulary")}, indent=2, sort_keys=True))
    print(f"wrote {out_csv}")
    return 0


def write_tables(summary: dict) -> None:
    lines = ["# Kekule support tables", ""]
    v = summary["vocabulary"]
    lines += ["## Train vocabulary sizes", "",
              "| vocabulary | types |", "|---|---:|",
              f"| orig | {v['n_orig_types']} |",
              f"| rep (orbit quotient) | {v['n_rep_types']} |",
              f"| arom (ordinary aromaticisation) | {v['n_arom_types']} |",
              f"| union (joint statistics) | {v['n_union_types']} |", ""]
    lines += ["## Merge / split accounting (train occurrences)", "",
              "| statistic | value |", "|---|---:|",
              f"| atom occurrences | {v['n_occurrences']} |",
              f"| occurrences with >1 legal key | {v['n_multi_orbit_occurrences']} |",
              f"| orig keys with >1 rep image (context split) | {v['orig_keys_with_multiple_rep_images']} |",
              f"| rep merge classes | {v['rep_merge_classes']} |",
              f"| rep merged extra original keys | {v['rep_merged_extra_keys']} |",
              f"| rep classes with >1 witness component | {v['rep_merged_classes_with_multiple_witness_components']} |",
              f"| rep unwitnessed merged extra keys | {v['rep_unwitnessed_merged_extra_keys']} |",
              f"| orig keys with >1 arom image (context split) | {v['orig_keys_with_multiple_arom_images']} |",
              f"| arom merge classes | {v['arom_merge_classes']} |",
              f"| arom merged extra original keys | {v['arom_merged_extra_keys']} |",
              f"| arom classes with >1 witness component | {v['arom_merged_classes_with_multiple_witness_components']} |",
              f"| arom unwitnessed merged extra keys | {v['arom_unwitnessed_merged_extra_keys']} |", ""]
    lines += ["## Group support", "",
              "| group | n | atoms | scheme | unseen atoms | pooled unseen frac | mean per-mol frac | fully supported |",
              "|---|---:|---:|---|---:|---:|---:|---:|"]
    for gname, block in summary["groups"].items():
        for s, d in block["schemes"].items():
            lines.append(
                f"| {gname} | {block['n_molecules']} | {block['n_atoms']} | {s} | "
                f"{d['unseen_atoms']} | {d['unseen_atom_fraction_pooled']:.6f} | "
                f"{d['unseen_atom_fraction_mean_per_molecule']:.6f} | "
                f"{d['fully_supported_molecules']}/{block['n_molecules']} |")
    lines += ["", "## H-inference audit", "",
              "| split | molecules with inferred H | explicit-H-only no solution | original missing | H-changing candidates |",
              "|---|---:|---:|---:|---:|"]
    for split in ("valid", "train"):
        h = summary["h_audit"][split]
        c = h["counts"]
        lines.append(f"| {split} | {h['n_molecules_with_inferred_h']} | "
                     f"{c.get('explicit_h_no_solution', 0)} | "
                     f"{c.get('explicit_h_original_missing', 0)} | "
                     f"{c.get('explicit_h_h_changing_candidates', 0)} |")
    lines.append("")
    (ANALYSIS / "support_tables.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
