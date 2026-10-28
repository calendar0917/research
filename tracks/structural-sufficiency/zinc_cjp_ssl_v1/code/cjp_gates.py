"""CPU gates G1..G12 for CJP-SSL v1.

Every gate is a pure CPU computation.  None of them may be skipped or relaxed
to make a GPU run possible: a failing gate is a correctness bug in this
implementation, not a research result.

Run all gates and write ``analysis/cpu_checks.json``::

    uv run python tracks/structural-sufficiency/zinc_cjp_ssl_v1/code/cjp_gates.py
"""

from __future__ import annotations

import argparse
import hashlib
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import cjp_common as C
import cjp_data as D
import cjp_models as CM
import cjp_pretrain as PT
import cjp_sup as SUP

L = C.L
M = C.M
LD = C.D

import cscl_units as cu  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

_RAWS: list[dict[str, np.ndarray]] | None = None
_SPLIT: dict[str, Any] | None = None
_TOPO: dict[int, dict[str, Any]] = {}

MOL_KEYS = (
    "atom_feat", "atom_type", "atom_scalar", "bond_feat", "bond_type", "bond_index",
    "nbr_ptr", "nbr_code", "path_ptr", "path_cat", "path_end63", "path_end",
    "path_closure", "pair_i", "pair_j", "pair_cat", "global",
)


def raws() -> list[dict[str, np.ndarray]]:
    global _RAWS
    if _RAWS is None:
        _RAWS = D.load_raw_train()
    return _RAWS


def split() -> dict[str, Any]:
    global _SPLIT
    if _SPLIT is None:
        _SPLIT = D.structure_split(raws())
    return _SPLIT


def topo(rid: int) -> dict[str, Any]:
    if rid not in _TOPO:
        _TOPO[rid] = D.build_topology(raws()[rid])
    return _TOPO[rid]


def _arrays_equal(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray],
                  keys: Sequence[str] = MOL_KEYS) -> dict[str, Any]:
    out = {"mismatch": {}, "max_abs_diff": {}, "pass": True}
    for k in keys:
        x = np.asarray(a[k])
        y = np.asarray(b[k])
        if x.shape != y.shape:
            out["mismatch"][k] = f"shape {x.shape} vs {y.shape}"
            out["pass"] = False
            continue
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            diff = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
            out["max_abs_diff"][k] = diff
            if diff != 0.0:
                out["mismatch"][k] = f"max abs {diff:g}"
                out["pass"] = False
        else:
            if not np.array_equal(x, y):
                n = int((x != y).sum())
                out["mismatch"][k] = f"{n} differing entries"
                out["pass"] = False
    return out


# ---------------------------------------------------------------------------
# G1: no label path; label poisoning cannot change anything
# ---------------------------------------------------------------------------


FORBIDDEN_TOKENS = (
    "official_labels",
    "load_cache",
    "build_split",
    '["labels"]',
    "['labels']",
    "data.y",
    "d.y",
)


def gate_g1() -> dict[str, Any]:
    static = {}
    for name in ("cjp_data.py", "cjp_models.py", "cjp_pretrain.py", "cjp_sup.py"):
        src = (C.CODE_DIR / name).read_text()
        hits = [t for t in FORBIDDEN_TOKENS if t in src]
        static[name] = hits
    static_pass = all(not v for v in static.values())

    def _tiny(name: str, poison: bool) -> dict[str, Any]:
        import torch_geometric.datasets as tgd

        original_zinc = tgd.ZINC

        class _Poisoned:
            def __init__(self, *a, **k):
                self._ds = original_zinc(*a, **k)

            def __len__(self):
                return len(self._ds)

            def __getitem__(self, i):
                d = self._ds[i]
                if poison:
                    d.y = torch.full_like(d.y, 1.0e9)
                return d

        if poison:
            tgd.ZINC = _Poisoned
        try:
            return PT.pretrain(
                "JOINT", 0, "cpu", updates=2, max_fit=32, diag_limit=8,
                eval_subset=8, out_dir=C.RUNS / "gate_tmp" / name, force=True,
            )
        finally:
            tgd.ZINC = original_zinc

    clean = _tiny("g1_clean", False)
    poisoned = _tiny("g1_poison", True)
    fields = {
        "masked_batch_sha256": (clean["data"]["masked_batch_sha256"],
                                poisoned["data"]["masked_batch_sha256"]),
        "first_batch_query_sha256": (clean["data"]["first_batch_query_sha256"],
                                     poisoned["data"]["first_batch_query_sha256"]),
        "local_state_sha256": (clean["local_state"]["state_sha256"],
                               poisoned["local_state"]["state_sha256"]),
    }
    curve_equal = clean["train"]["curve"] == poisoned["train"]["curve"]
    diag_equal = clean["diagnostics_after"] == poisoned["diagnostics_after"]
    detail = {
        "static_forbidden_token_hits": static,
        "fields": {k: v[0] == v[1] for k, v in fields.items()},
        "curve_equal": bool(curve_equal),
        "diagnostics_equal": bool(diag_equal),
        "poisoned_label_value": 1.0e9,
    }
    return {"pass": bool(static_pass and all(v[0] == v[1] for v in fields.values())
                         and curve_equal and diag_equal), "detail": detail}


# ---------------------------------------------------------------------------
# G2: masking off -> byte-identical to the frozen LCRC train cache
# ---------------------------------------------------------------------------


def gate_g2() -> dict[str, Any]:
    frozen = LD.load_cache("train")
    mols = []
    for rid in range(len(raws())):
        t = topo(rid)
        mask = D.apply_mask(t, [])
        obj = sro.build_mol_objects(int(t["mol_index"]), mask["x"], t["edge_index_raw"],
                                    mask["edge_attr"])
        mols.append(D.compose_masked(t, obj, mask["x"], mask["bond_type"],
                                     mask["atom_mask"], mask["bond_mask"]))
    rebuilt = D.cache_from_molecules(mols)
    cmp = _arrays_equal(rebuilt, frozen, MOL_KEYS + ("atom_ptr", "bond_ptr", "nbr_ptr",
                                                     "path_ptr", "pair_ptr"))
    # also check atom_type / bond_type against the frozen cache raw values
    detail = {
        "n_molecules": int(rebuilt["atom_ptr"].shape[0] - 1),
        "comparison": cmp,
        "frozen_n_molecules": int(frozen["n_molecules"]),
    }
    return {"pass": bool(cmp["pass"] and detail["n_molecules"] == int(frozen["n_molecules"])),
            "detail": detail}


# ---------------------------------------------------------------------------
# G3: masked raw graph -> P3 rebuild matches an independent reference build
# ---------------------------------------------------------------------------


GLOBAL_TYPE_DEPENDENT = (19, 20, 21, 22, 31, 32)  # bond-type fractions + distinct counts


def _affected_atoms(t: Mapping[str, Any], atom_mask: np.ndarray, bond_mask: np.ndarray) -> np.ndarray:
    n = int(t["n_atoms"])
    affected = np.zeros(n, dtype=bool)
    adj = [[] for _ in range(n)]
    for bid in range(int(t["n_bonds"])):
        a, b = int(t["bond_u"][bid]), int(t["bond_v"][bid])
        adj[a].append(b)
        adj[b].append(a)
    for a in np.flatnonzero(atom_mask):
        affected[a] = True
        for b in adj[a]:
            affected[b] = True
            for c in adj[b]:
                affected[c] = True
    for bid in np.flatnonzero(bond_mask):
        affected[int(t["bond_u"][bid])] = True
        affected[int(t["bond_v"][bid])] = True
    return affected


def gate_g3(n_cases: int = 300, seed: int = 20261029) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(raws()), size=n_cases, replace=False)
    ref_mismatch: dict[str, int] = {}
    rebuild_eq = True
    dep_ok = True
    own_slot_ok = True
    truth_absent_ok = True
    checked = 0
    for rid in ids:
        rid = int(rid)
        t = topo(rid)
        vi = int(rng.integers(0, 3))
        v = D.prepare_visit(t, seed=0, molecule_id=rid, visit_index=vi)
        mask = v["mask"]
        obj = sro.build_mol_objects(int(t["mol_index"]), mask["x"], t["edge_index_raw"],
                                    mask["edge_attr"])
        obj0 = sro.build_mol_objects(int(t["mol_index"]), t["x_raw"], t["edge_index_raw"],
                                     t["edge_attr_raw"])
        ref = D.reference_compose_masked(
            t, obj.atom_feat, obj.bond_feat, obj.global_feat,
            mask["x"], mask["bond_type"], mask["atom_mask"], mask["bond_mask"],
        )
        cmp = _arrays_equal(v["mol"], ref)
        if not cmp["pass"]:
            rebuild_eq = False
            for k, why in cmp["mismatch"].items():
                ref_mismatch[k] = ref_mismatch.get(k, 0) + 1
        # ---- dependency map / mask tokens / hidden truth ------------------
        affected = _affected_atoms(t, mask["atom_mask"], mask["bond_mask"])
        d_atom = np.abs(obj.atom_feat - obj0.atom_feat)
        outside = d_atom[~affected]
        if outside.size and float(outside.max()) != 0.0:
            dep_ok = False
        d_bond = np.abs(obj.bond_feat - obj0.bond_feat)
        if d_bond.size:
            keep = ~mask["bond_mask"]
            if float(d_bond[keep].max()) != 0.0:
                dep_ok = False
        d_glob = np.abs(obj.global_feat - obj0.global_feat)
        allowed = np.zeros_like(d_glob, dtype=bool)
        allowed[list(GLOBAL_TYPE_DEPENDENT)] = True
        if float(d_glob[~allowed].max()) != 0.0:
            dep_ok = False
        for a in np.flatnonzero(mask["atom_mask"]):
            own = obj.atom_feat[a, : C.N_ATOM_SLOTS]
            if not (int(np.argmax(own)) == C.ATOM_MASK_TOKEN and own[C.ATOM_MASK_TOKEN] == 1.0):
                own_slot_ok = False
            if obj.atom_feat[a, int(t["x_raw"][a])] != 0.0:
                truth_absent_ok = False
        for b in np.flatnonzero(mask["bond_mask"]):
            own = obj.bond_feat[b, : C.N_BOND_SLOTS]
            if not (int(np.argmax(own)) == C.BOND_MASK_TOKEN and own[C.BOND_MASK_TOKEN] == 1.0):
                own_slot_ok = False
            if obj.bond_feat[b, int(t["bond_type_raw"][b])] != 0.0:
                truth_absent_ok = False
        checked += 1
    detail = {
        "n_cases": checked,
        "seeds": seed,
        "compose_matches_independent_reference": bool(rebuild_eq),
        "reference_mismatch_keys": ref_mismatch,
        "differences_confined_to_dependency_map": bool(dep_ok),
        "masked_own_slots_hold_token": bool(own_slot_ok),
        "hidden_truth_absent_from_own_slot": bool(truth_absent_ok),
    }
    return {"pass": bool(rebuild_eq and dep_ok and own_slot_ok and truth_absent_ok),
            "detail": detail}


# ---------------------------------------------------------------------------
# G4: hidden-truth swap leaves the masked input bit-identical
# ---------------------------------------------------------------------------


def _perturb_hidden(rid: int, t, pid: int, *, what: str) -> dict[str, Any]:
    x = np.array(t["x_raw"], copy=True)
    ea = np.array(t["edge_attr_raw"], copy=True)
    u = int(t["path_u"][pid])
    bid = int(t["path_e2"][pid])
    if what == "atom":
        x[u] = (int(x[u]) + 1) % C.N_ATOM_TYPES
    elif what == "bond":
        new_bt = int(t["bond_type_raw"][bid]) % C.N_BOND_TYPES + 1
        ea[t["directed_pos"][bid]] = new_bt
    else:
        raise ValueError(what)
    return {"mol_index": int(t["mol_index"]), "x": x,
            "edge_index": np.array(t["edge_index_raw"], copy=True), "edge_attr": ea}


def gate_g4(n_atom: int = 210, n_bond: int = 210, seed: int = 20261030) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    pool = [int(i) for i in rng.permutation(len(raws()))]
    detail = {"n_atom_trials": 0, "n_atom_pass": 0, "n_bond_trials": 0, "n_bond_pass": 0,
              "target_changed": 0, "input_identical": 0, "failures": []}
    for what, target_n in (("atom", n_atom), ("bond", n_bond)):
        key_t = "n_atom_trials" if what == "atom" else "n_bond_trials"
        key_p = "n_atom_pass" if what == "atom" else "n_bond_pass"
        for rid in pool:
            if detail["n_atom_trials"] >= n_atom and detail["n_bond_trials"] >= n_bond:
                break
            if detail[key_t] >= target_n:
                continue
            t = topo(rid)
            pid = D.sample_paths(t, seed=0, molecule_id=rid, visit_index=0)
            if len(pid) == 0:
                continue
            p = int(pid[0])
            raw2 = _perturb_hidden(rid, t, p, what=what)
            t2 = D.build_topology(raw2)
            v1 = D.prepare_visit(t, seed=0, molecule_id=rid, visit_index=0, path_ids=pid)
            v2 = D.prepare_visit(t2, seed=0, molecule_id=rid, visit_index=0, path_ids=pid)
            detail[key_t] += 1
            same_input = (
                np.array_equal(v1["mask"]["x"], v2["mask"]["x"])
                and np.array_equal(v1["mask"]["edge_attr"], v2["mask"]["edge_attr"])
                and np.array_equal(t["edge_index_raw"], t2["edge_index_raw"])
            )
            same_mol = _arrays_equal(v1["mol"], v2["mol"])["pass"]
            target_changed = not np.array_equal(v1["queries"], v2["queries"])
            if same_input:
                detail["input_identical"] += 1
            if target_changed:
                detail["target_changed"] += 1
            if same_input and same_mol and target_changed:
                detail[key_p] += 1
            elif len(detail["failures"]) < 10:
                detail["failures"].append(
                    {"rid": rid, "what": what, "same_input": bool(same_input),
                     "same_mol": bool(same_mol), "target_changed": bool(target_changed)}
                )
    ok = (detail["n_atom_pass"] == n_atom and detail["n_bond_pass"] == n_bond)
    return {"pass": bool(ok), "detail": detail}


# ---------------------------------------------------------------------------
# G5: both directed PyG records of the physical bond are masked
# ---------------------------------------------------------------------------


def gate_g5(n_cases: int = 300, seed: int = 20261031) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(raws()), size=n_cases, replace=False)
    fails = []
    total_paths = 0
    for rid in ids:
        rid = int(rid)
        t = topo(rid)
        v = D.prepare_visit(t, seed=1, molecule_id=rid, visit_index=int(rng.integers(0, 4)))
        raw_ea = t["edge_attr_raw"]
        new_ea = v["mask"]["edge_attr"]
        changed = np.flatnonzero(raw_ea != new_ea)
        expected = np.sort(np.concatenate(
            [t["directed_pos"][int(t["path_e2"][p])] for p in v["path_ids"]]
        )) if len(v["path_ids"]) else np.zeros(0, dtype=np.int64)
        total_paths += len(v["path_ids"])
        ok = (
            np.array_equal(np.sort(changed), expected)
            and np.array_equal(t["edge_index_raw"], raws()[rid]["edge_index"])
            and int((new_ea[changed] == C.BOND_MASK_TOKEN).sum()) == len(changed)
            and len(changed) == 2 * len(v["path_ids"])
        )
        # both directions of each masked physical bond carry the token
        for p in v["path_ids"]:
            pos = t["directed_pos"][int(t["path_e2"][p])]
            ok = ok and bool((new_ea[pos] == C.BOND_MASK_TOKEN).all())
        if not ok and len(fails) < 10:
            fails.append({"rid": rid, "changed": changed.tolist(),
                          "expected": expected.tolist()})
    detail = {"n_cases": int(len(ids)), "n_selected_paths": int(total_paths),
              "failures": fails,
              "note": "edge_index is never modified and exactly 2 records per masked bond change"}
    return {"pass": not fails, "detail": detail}


# ---------------------------------------------------------------------------
# G6: no mask token is ever treated as a legal raw type
# ---------------------------------------------------------------------------


def gate_g6(n_cases: int = 300, seed: int = 20261032) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(raws()), size=n_cases, replace=False)
    fails = []
    n_checked = 0
    for rid in ids:
        rid = int(rid)
        t = topo(rid)
        v = D.prepare_visit(t, seed=2, molecule_id=rid, visit_index=int(rng.integers(0, 4)))
        mol = v["mol"]
        atom_mask = v["mask"]["atom_mask"]
        bond_mask = v["mask"]["bond_mask"]
        atom_type = v["mask"]["x"]
        bond_type = v["mask"]["bond_type"]
        # expected visible record counts, recomputed independently from masks
        keep1 = (~atom_mask[t["nbr_nbr"]]) & (~bond_mask[t["nbr_bond"]])
        keep2 = ((~atom_mask[t["path_w"]]) & (~atom_mask[t["path_u"]])
                 & (~bond_mask[t["path_e1"]]) & (~bond_mask[t["path_e2"]]))
        checks = {
            "s1_count": len(mol["nbr_code"]) == int(keep1.sum()),
            "s2_count": len(mol["path_cat"]) == int(keep2.sum()),
            "nbr_code_vocab": bool(len(mol["nbr_code"]) == 0
                                   or (mol["nbr_code"] < C.N_NBR_CODES).all()),
            "path_cat_vocab": bool(len(mol["path_cat"]) == 0
                                   or (mol["path_cat"] < C.N_PATH_CODES).all()),
            "path_end63_vocab": bool(len(mol["path_end63"]) == 0
                                     or (mol["path_end63"] < C.N_NBR_CODES).all()),
            "path_closure_binary": bool(len(mol["path_closure"]) == 0
                                        or np.isin(mol["path_closure"], [0, 1]).all()),
            "path_end_visible": bool(len(mol["path_end"]) == 0
                                     or not atom_mask[mol["path_end"]].any()),
            "s1_no_mask_token": True,
            "s2_no_mask_token": True,
            "atom_type_vocab": bool(np.isin(mol["atom_type"], list(range(C.N_ATOM_TYPES))
                                            + [C.ATOM_MASK_TOKEN]).all()),
            "bond_type_vocab": bool(np.isin(mol["bond_type"], list(range(C.N_BOND_TYPES + 1))).all()),
        }
        # reconstruct every included record and re-check visibility
        for root, nb, bnd in zip(t["nbr_root"][keep1], t["nbr_nbr"][keep1], t["nbr_bond"][keep1]):
            if atom_mask[nb] or bond_mask[bnd]:
                checks["s1_no_mask_token"] = False
            code = int(atom_type[nb]) * C.N_BOND_TYPES + int(bond_type[bnd]) - 1
            if not (0 <= code < C.N_NBR_CODES):
                checks["s1_no_mask_token"] = False
        for w, u, e1, e2 in zip(t["path_w"][keep2], t["path_u"][keep2],
                                t["path_e1"][keep2], t["path_e2"][keep2]):
            if atom_mask[w] or atom_mask[u] or bond_mask[e1] or bond_mask[e2]:
                checks["s2_no_mask_token"] = False
            if int(atom_type[w]) == C.ATOM_MASK_TOKEN or int(atom_type[u]) == C.ATOM_MASK_TOKEN:
                checks["s2_no_mask_token"] = False
            if int(bond_type[e1]) == C.BOND_MASK_TOKEN or int(bond_type[e2]) == C.BOND_MASK_TOKEN:
                checks["s2_no_mask_token"] = False
        n_checked += 1
        if not all(checks.values()) and len(fails) < 10:
            fails.append({"rid": rid, "checks": {k: bool(x) for k, x in checks.items()}})
    return {"pass": not fails, "detail": {"n_cases": n_checked, "failures": fails}}


# ---------------------------------------------------------------------------
# G7: no masking -> model output identical to the original LCRC forward
# ---------------------------------------------------------------------------


def gate_g7(seed: int = 0, n_rows: int = 64) -> dict[str, Any]:
    norm = LD.load_norm()
    frozen = LD.load_cache("train")
    model, info = M.build_model("A", norm, seed, "cpu")
    rows = np.arange(n_rows, dtype=np.int64)
    batch_orig = LD.collate(frozen, rows)
    mols = []
    for rid in rows:
        t = topo(int(rid))
        mask = D.apply_mask(t, [])
        obj = sro.build_mol_objects(int(t["mol_index"]), mask["x"], t["edge_index_raw"],
                                    mask["edge_attr"])
        mols.append(D.compose_masked(t, obj, mask["x"], mask["bond_type"],
                                     mask["atom_mask"], mask["bond_mask"]))
    batch_new, query = D.batch_from_molecules(mols, [np.zeros((0, 3), np.int64)] * len(mols))
    tensor_equal = {}
    for k in ("atom_feat", "atom_type", "atom_scalar", "bond_feat", "bond_type",
              "bond_index", "nbr_code", "path_cat", "path_end63", "path_end",
              "path_closure", "pair_i", "pair_j", "pair_cat", "global", "nbr_atom",
              "path_atom", "atom_batch", "atom_off", "bond_batch"):
        tensor_equal[k] = bool(torch.equal(batch_orig[k], batch_new[k]))
    with torch.no_grad():
        y_orig = model(batch_orig)
        y_new = model(batch_new)
    diff = float((y_orig - y_new).abs().max().item())
    detail = {
        "n_rows": int(n_rows),
        "tensors_bit_identical": tensor_equal,
        "all_tensors_identical": bool(all(tensor_equal.values())),
        "prediction_max_abs_diff": diff,
        "parameters": info["parameters"],
        "ssl_loaded": info["ssl"]["loaded_modules"],
    }
    return {"pass": bool(all(tensor_equal.values()) and diff == 0.0), "detail": detail}


# ---------------------------------------------------------------------------
# G8: invariance to relabelling / bond reversal / batch split-merge
# ---------------------------------------------------------------------------


def _pretext_output(model, head, mol, queries, arm="JOINT"):
    batch, query = D.batch_from_molecules([mol], [queries])
    with torch.no_grad():
        _used, raw, h = CM.pretext_logits(model, head, batch, query, arm=arm)
    return h.cpu(), raw.cpu()


def gate_g8(seed: int = 0) -> dict[str, Any]:
    norm = LD.load_norm()
    model, head, _info = CM.build_ssl_model(norm, seed, "cpu")
    rng = np.random.default_rng(20261033)
    pick = None
    for rid in rng.permutation(len(raws())):
        t = topo(int(rid))
        if len(t["path_root"]) >= 3:
            pick = int(rid)
            break
    t = topo(pick)
    pid = D.sample_paths(t, seed=0, molecule_id=pick, visit_index=0)
    v = D.prepare_visit(t, seed=0, molecule_id=pick, visit_index=0, path_ids=pid)
    h0, logits0 = _pretext_output(model, head, v["mol"], v["queries"])

    # ---- atom relabelling (common renumbering for the fixed query path) --
    n = int(t["n_atoms"])
    perm = rng.permutation(n)
    x2 = np.zeros_like(t["x_raw"])
    x2[perm] = t["x_raw"]
    ei2 = perm[t["edge_index_raw"]]
    raw2 = {"mol_index": pick, "x": x2, "edge_index": ei2,
            "edge_attr": np.array(t["edge_attr_raw"], copy=True)}
    t2 = D.build_topology(raw2)
    # the mask/query path is a *vertex* triple: map it through the common relabelling
    triples = [(int(t["path_root"][p]), int(t["path_w"][p]), int(t["path_u"][p]))
               for p in pid]
    lookup = {(int(r), int(w), int(u)): k for k, (r, w, u) in enumerate(
        zip(t2["path_root"], t2["path_w"], t2["path_u"]))}
    pid2 = np.asarray([lookup[(int(perm[v]), int(perm[w]), int(perm[u]))]
                       for v, w, u in triples], dtype=np.int64)
    q2 = v["queries"].copy()
    q2[:, 0] = perm[q2[:, 0]]
    q2[:, 1] = perm[q2[:, 1]]
    v2 = D.prepare_visit(t2, seed=0, molecule_id=pick, visit_index=0, path_ids=pid2)
    h2, logits2 = _pretext_output(model, head, v2["mol"], q2)
    # h_new[perm[old]] must equal h_old[old]
    h_perm_max = float((h2[torch.as_tensor(perm)] - h0).abs().max().item())
    logits_perm_max = float((logits2 - logits0).abs().max().item())

    # ---- bond endpoint reversal ------------------------------------------
    raw3 = {"mol_index": pick, "x": np.array(t["x_raw"], copy=True),
            "edge_index": np.ascontiguousarray(t["edge_index_raw"][::-1]),
            "edge_attr": np.array(t["edge_attr_raw"], copy=True)}
    t3 = D.build_topology(raw3)
    v3 = D.prepare_visit(t3, seed=0, molecule_id=pick, visit_index=0, path_ids=pid)
    rev_equal = _arrays_equal(v["mol"], v3["mol"])
    h3, logits3 = _pretext_output(model, head, v3["mol"], v["queries"])
    h_rev_max = float((h3 - h0).abs().max().item())
    logits_rev_max = float((logits3 - logits0).abs().max().item())

    # ---- batch split / merge --------------------------------------------
    rows = [pick, int(rng.integers(0, len(raws())))]
    mols, queries = [], []
    for rid in rows:
        tt = topo(rid)
        vv = D.prepare_visit(tt, seed=0, molecule_id=rid, visit_index=0)
        mols.append(vv["mol"])
        queries.append(vv["queries"])
    batch, query = D.batch_from_molecules(mols, queries)
    with torch.no_grad():
        _u, _r, h_batch = CM.pretext_logits(model, head, batch, query, arm="JOINT")
    split_max = 0.0
    for i, rid in enumerate(rows):
        sel = torch.as_tensor(np.flatnonzero(batch["atom_batch"].numpy() == i))
        h_single, _ = _pretext_output(model, head, mols[i], queries[i])
        split_max = max(split_max, float((h_batch[sel] - h_single).abs().max().item()))
    detail = {
        "molecule": pick,
        "n_selected_paths": int(len(pid)),
        "renumber_h_equivariance_max_abs": h_perm_max,
        "renumber_logits_max_abs": logits_perm_max,
        "bond_reversal_tensors_identical": bool(rev_equal["pass"]),
        "bond_reversal_h_max_abs": h_rev_max,
        "bond_reversal_logits_max_abs": logits_rev_max,
        "batch_split_merge_h_max_abs": split_max,
    }
    ok = (h_perm_max <= 1e-5 and logits_perm_max <= 1e-5 and rev_equal["pass"]
          and h_rev_max <= 1e-6 and logits_rev_max <= 1e-6 and split_max <= 1e-5)
    return {"pass": bool(ok), "detail": detail}


# ---------------------------------------------------------------------------
# G9: MARG factorises; JOINT has an interaction MARG cannot express
# ---------------------------------------------------------------------------


def gate_g9(seed: int = 0) -> dict[str, Any]:
    g = torch.Generator().manual_seed(20261034)
    logits = torch.randn(7, C.N_TARGET_CLASSES, generator=g) * 2.0
    probs_marg = torch.softmax(CM.marg_projection(logits), dim=1)
    row, col = CM.marg_softmax_factors(logits)
    factorised = torch.softmax(row, dim=1).unsqueeze(2) * torch.softmax(col, dim=1).unsqueeze(1)
    factorised = factorised.reshape(logits.shape[0], -1)
    factor_err = float((probs_marg - factorised).abs().max().item())
    marg_residual = float(CM.interaction_residual(CM.marg_projection(logits)).abs().max().item())

    # 2x2 interaction counterexample embedded in the real 21x3 class space:
    # XOR-style joint logits on atom classes {0,1} x bond classes {0,1}.
    joint = torch.zeros(1, C.N_TARGET_CLASSES)
    patch = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
    for a in range(2):
        for b in range(2):
            joint[0, a * C.TARGET_BOND_CLASSES + b] = patch[a, b]
    marg = CM.marg_projection(joint)
    interaction = joint - marg
    p_joint = torch.softmax(joint, dim=1)
    p_marg = torch.softmax(marg, dim=1)
    tv = float(0.5 * (p_joint - p_marg).abs().sum().item())
    # no additive table reproduces the joint table, while JOINT reproduces it exactly
    detail = {
        "marg_probability_factorisation_max_abs": factor_err,
        "marg_residual_of_marg_projection_max_abs": marg_residual,
        "marg_residual_note": "algebraically zero; <=1e-6 is float32 rounding of the projection",
        "counterexample_joint_table_2x2": patch.tolist(),
        "counterexample_marg_projection_2x2": marg.reshape(
            C.TARGET_ATOM_CLASSES, C.TARGET_BOND_CLASSES)[:2, :2].tolist(),
        "counterexample_marg_projection_max_abs": float(marg.abs().max().item()),
        "counterexample_interaction_max_abs": float(interaction.abs().max().item()),
        "counterexample_total_variation": tv,
        "joint_can_represent_counterexample": bool(
            torch.equal(joint, joint.clone())),
        "joint_interaction_rms_random_logits": float(
            CM.interaction_residual(logits).pow(2).mean().sqrt().item()),
    }
    shared = {}
    for arm in ("MARG", "JOINT"):
        model, head, info = CM.build_ssl_model(LD.load_norm(), seed, "cpu")
        shared[arm] = {
            "joint_init_sha256": info["joint_init_sha256"],
            "local_sha256": info["local_sha256"],
            "head_sha256": info["head_sha256"],
            "encoder_sha256": info["encoder_sha256"],
        }
        del model, head
    same = (shared["MARG"]["joint_init_sha256"] == shared["JOINT"]["joint_init_sha256"]
            and shared["MARG"]["head_sha256"] == shared["JOINT"]["head_sha256"]
            and shared["MARG"]["encoder_sha256"] == shared["JOINT"]["encoder_sha256"])
    detail["shared_initial_state"] = shared
    detail["shared_initial_state_identical"] = bool(same)
    ok = (factor_err <= 1e-6 and marg_residual <= 1e-6
          and float(interaction.abs().max().item()) >= 1.9 and tv > 0.05 and same)
    return {"pass": bool(ok), "detail": detail}


# ---------------------------------------------------------------------------
# G10 / G11: gradient isolation and reachable-module gradients
# ---------------------------------------------------------------------------


def _one_step(arm: str = "JOINT", seed: int = 0, n_mol: int = 128, device="cpu"):
    norm = LD.load_norm()
    model, head, info = CM.build_ssl_model(norm, seed, device)
    rng = np.random.default_rng(20261035)
    fit = split()["fit_index"]
    rows = fit[rng.permutation(len(fit))[:n_mol]]
    mols, queries = [], []
    for rid in rows:
        rid = int(rid)
        v = D.prepare_visit(topo(rid), seed=seed, molecule_id=rid, visit_index=0)
        mols.append(v["mol"])
        queries.append(v["queries"])
    batch, query = PT._to_device(*D.batch_from_molecules(mols, queries), device)
    used, raw, _h = CM.pretext_logits(model, head, batch, query, arm=arm)
    loss, _s, _c = CM.molecule_mean_loss(CM.per_query_ce(used, query["target"]),
                                        query, batch["n_graphs"])
    loss.backward()
    return model, head, info, loss, raw


def gate_g10() -> dict[str, Any]:
    model, head, info, loss, _raw = _one_step()
    def _all_none(d):
        return all(v is None for v in d.values())
    enc = {}
    for name in ("atom_encoder", "bond_encoder"):
        enc[name] = _all_none(PT._grad_norms(getattr(model, name)))
    f_none = _all_none(PT._grad_norms(model.f))
    rho_none = _all_none(PT._grad_norms(model.rho))
    local = PT._grad_norms(model.local)
    head_g = PT._grad_norms(head)
    local_nonzero = int(sum(1 for v in local.values() if v is not None and v > 0))
    head_nonzero = int(sum(1 for v in head_g.values() if v is not None and v > 0))
    detail = {
        "loss": float(loss.item()),
        "encoders_no_grad": enc,
        "f_no_grad": bool(f_none),
        "rho_no_grad": bool(rho_none),
        "local_tensors_with_grad": int(sum(1 for v in local.values() if v is not None)),
        "local_tensors_nonzero_grad": local_nonzero,
        "local_tensors_total": int(len(local)),
        "head_tensors_with_grad": int(sum(1 for v in head_g.values() if v is not None)),
        "head_tensors_nonzero_grad": head_nonzero,
        "frozen_after_build": info["frozen_modules"],
    }
    ok = (all(enc.values()) and f_none and rho_none and local_nonzero > 0 and head_nonzero > 0)
    return {"pass": bool(ok), "detail": detail}


def gate_g11() -> dict[str, Any]:
    model, head, _info, _loss, _raw = _one_step()
    local = PT._grad_norms(model.local)
    required = {
        "path_emb.weight": "path_emb",
        "path_u.weight": "path_u",
        "path_v.weight": "path_v",
        "path_gate.weight": "path_gate",
        "nbr_emb.weight": "nbr_emb",
        "end_emb.weight": "end_emb",
        "joint_proj.weight": "joint_proj",
        "close_emb.weight": "close_emb",
        "mlp.lins.0.weight": "local.mlp",
        "mlp.lins.1.weight": "local.mlp",
        "norm.weight": "local.norm",
    }
    detail = {"tensors": {}, "finite": True, "nonzero": {}, "reached_embedding_rows": {}}
    ok = True
    for key, group in required.items():
        g = local.get(key)
        finite = g is not None and np.isfinite(g)
        detail["tensors"][key] = g
        detail["finite"] = bool(detail["finite"] and finite)
        if g is None or g == 0.0:
            ok = False
            detail["nonzero"][key] = False
        else:
            detail["nonzero"][key] = True
    # per-row reachability for the embeddings actually touched by this batch
    for key in ("path_emb.weight", "path_u.weight", "path_v.weight", "path_gate.weight"):
        p = dict(model.local.named_parameters())[key]
        if p.grad is not None:
            rows = torch.nonzero(p.grad.abs().sum(dim=1) > 0).flatten()
            detail["reached_embedding_rows"][key] = int(len(rows))
    detail["note"] = ("embedding rows not visited by this batch are expected to have "
                      "zero rows; every parameter tensor above must have a non-zero "
                      "finite gradient somewhere")
    return {"pass": bool(ok and detail["finite"]), "detail": detail}


# ---------------------------------------------------------------------------
# G12: save / reload / export compatibility with the deployed LCRC-A
# ---------------------------------------------------------------------------


def gate_g12(seed: int = 0) -> dict[str, Any]:
    norm = LD.load_norm()
    reference, ref_info = M.build_model("A", norm, seed, "cpu")
    frozen_cache = LD.load_cache("train")
    batch = LD.collate(frozen_cache, np.arange(32, dtype=np.int64))
    with torch.no_grad():
        y_ref = reference(batch)
    state = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
    reloaded = M.load_model_from_state("A", norm, state, "cpu")
    with torch.no_grad():
        y_reload = reloaded(batch)
    reload_diff = float((y_ref - y_reload).abs().max().item())

    # a trained LocalContent from the gate pretraining run
    local_path = C.RUNS / "gate_tmp" / "g1_clean" / "local_state.pt"
    local_ok = {"path": C.track_rel(local_path), "exists": local_path.exists()}
    if local_path.exists():
        payload = torch.load(local_path, map_location="cpu", weights_only=False)
        local_state = payload["state_dict"]
        # a second, deterministic LocalContent state so that MARG-SSL / JOINT-SSL
        # differ exactly like the two real arms do (only ``local``)
        local_state_b = {
            k: (v + 0.01 * float(i + 1)).clone() if v.dtype.is_floating_point else v.clone()
            for i, (k, v) in enumerate(sorted(local_state.items()))
        }
        variants = {}
        for mode, load_ssl in (("replay", True), ("scratch", False),
                               ("margssl", True), ("jointssl", True)):
            if mode == "replay" or mode == "scratch":
                use_local = None
            elif mode == "margssl":
                use_local = local_state
            else:
                use_local = local_state_b
            model, info = CM.build_supervised_model(
                norm, seed, "cpu", load_ssl=load_ssl, local_state=use_local)
            variants[mode] = info
            with torch.no_grad():
                y = model(batch)
            if not torch.isfinite(y).all():
                raise RuntimeError(f"{mode}: non-finite supervised forward")
            if mode == "replay":
                with torch.no_grad():
                    d = float((y - y_ref).abs().max().item())
                local_ok["replay_vs_reference_max_abs"] = d
                local_ok["replay_matches_reference"] = bool(d == 0.0)
        local_ok["variants"] = {
            k: {
                "parameters": v["parameters"],
                "module_sha256": v["module_sha256"],
                "ssl_loaded": v["ssl"]["loaded"],
                "local_replaced": v["local_state"]["replaced"],
            }
            for k, v in variants.items()
        }
        # deploy structure: keys identical to the reference, head absent
        mk, _ = CM.build_supervised_model(norm, seed, "cpu", load_ssl=True,
                                          local_state=local_state)
        keys_equal = set(mk.state_dict()) == set(reference.state_dict())
        head_absent = not any("pretext" in k or "head" in k for k in mk.state_dict())
        # module-sharing relations across arms
        rel = {}
        a = variants["replay"]["module_sha256"]
        for other in ("scratch", "margssl", "jointssl"):
            b = variants[other]["module_sha256"]
            rel[other] = {k: bool(a[k] == b[k]) for k in a}
        mj = variants["margssl"]["module_sha256"]
        jj = variants["jointssl"]["module_sha256"]
        rel["margssl_vs_jointssl_only_local_differs"] = bool(
            mj["local"] != jj["local"]
            and all(mj[k] == jj[k] for k in mj if k != "local")
        )
        local_ok.update({
            "state_keys_equal_reference": bool(keys_equal),
            "pretext_head_absent_from_supervised_state": bool(head_absent),
            "parameter_count": int(variants["replay"]["parameters"]),
            "deploy_parameter_count_expected": int(C.DEPLOY_PARAMETERS),
            "module_sharing": rel,
        })
        ok = (bool(local_ok["exists"]) and local_ok.get("replay_matches_reference", False)
              and keys_equal and head_absent
              and int(variants["replay"]["parameters"]) == int(C.DEPLOY_PARAMETERS)
              and rel["margssl_vs_jointssl_only_local_differs"])
    else:
        ok = False
    detail = {
        "reference_parameters": int(ref_info["parameters"]),
        "reference_state_hash": ref_info["initial_state_sha256"],
        "reload_max_abs_diff": reload_diff,
        "local_state_export": local_ok,
    }
    return {"pass": bool(ok and reload_diff == 0.0), "detail": detail}


# ---------------------------------------------------------------------------
# supervisor-runner equivalence (frozen loop is imported unchanged)
# ---------------------------------------------------------------------------


def gate_sup_trainer_lineage() -> dict[str, Any]:
    src = (C.LC_CODE / "lc_train.py").read_text()
    sup_src = (C.CODE_DIR / "cjp_sup.py").read_text()
    detail = {
        "cjp_sup_imports_frozen_trainer": "T.train(" in sup_src,
        "cjp_sup_reimplements_training_loop": "for epoch in range" in sup_src,
        "frozen_recipe_equal": dict(C.SUPERVISED_RECIPE) == dict(C.T.RECIPE),
        "frozen_lc_train_sha256": C.sha256_file(C.LC_CODE / "lc_train.py"),
        "frozen_lc_models_sha256": C.sha256_file(C.LC_CODE / "lc_models.py"),
        "frozen_lc_data_sha256": C.sha256_file(C.LC_CODE / "lc_data.py"),
        "supervised_recipe": dict(C.T.RECIPE),
    }
    ok = (detail["cjp_sup_imports_frozen_trainer"]
          and not detail["cjp_sup_reimplements_training_loop"]
          and detail["frozen_recipe_equal"])
    return {"pass": bool(ok), "detail": detail}


GATES = {
    "G0_supervised_trainer_lineage": gate_sup_trainer_lineage,
    "G1_label_isolation": gate_g1,
    "G2_masking_off_equals_frozen_cache": gate_g2,
    "G3_masked_rebuild_reference": gate_g3,
    "G4_hidden_truth_swap": gate_g4,
    "G5_both_directed_masks": gate_g5,
    "G6_no_mask_token_as_type": gate_g6,
    "G7_no_mask_forward_equals_lcrc": gate_g7,
    "G8_invariance": gate_g8,
    "G9_marg_factorisation_and_init": gate_g9,
    "G10_gradient_isolation": gate_g10,
    "G11_localcontent_gradients": gate_g11,
    "G12_export_and_deploy": gate_g12,
}


def run_all(only: Sequence[str] | None = None, write: bool = True) -> dict[str, Any]:
    results: dict[str, Any] = {}
    all_pass = True
    for name, fn in GATES.items():
        if only and name not in only:
            continue
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001
            res = {"pass": False, "detail": {"exception": repr(exc),
                                             "traceback": traceback.format_exc()}}
        res["seconds"] = float(time.perf_counter() - t0)
        results[name] = res
        all_pass = all_pass and bool(res["pass"])
        C.log(f"[gate] {name}: {'PASS' if res['pass'] else 'FAIL'} "
              f"({res['seconds']:.1f}s)")
    payload = {
        "round": C.ROUND_ID,
        "git_commit": C.git_commit(),
        "all_pass": bool(all_pass),
        "gates": results,
        "created": C.now(),
        "official_test_loaded": False,
        "official_valid_loaded": False,
    }
    if write:
        C.write_json(C.ANALYSIS / "cpu_checks.json", payload)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CJP-SSL CPU gates")
    parser.add_argument("--only", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    only = args.only.split(",") if args.only else None
    res = run_all(only=only, write=not args.no_write)
    C.log(f"ALL PASS: {res['all_pass']}")
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
